import torch
from torch.utils.data import DataLoader, random_split
import matplotlib
import matplotlib.pyplot as plt
from rsl_rl.modules import plotter
import wandb

matplotlib.use('Agg')


class ModelTraining:
    def __init__(
        self,
        log_dir,
        history_horizon,
        forecast_horizon,
        dataset,
        system_dynamics,
        device,
        optimizer,
        batch_size=512,
        eval_traj_config=None,
        eval_traj_noise_scale=[0.1],
        system_dynamics_loss_weights={"state": 1.0, "sequence": 1.0, "bound": 1.0, "kl": 0.1, "contact": 1.0, "termination": 1.0},
        save_interval=200,
        max_iterations=1000,
        ):
        self.log_dir = log_dir
        self.history_horizon = history_horizon
        self.forecast_horizon = forecast_horizon
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        self.train_dataset, self.test_dataset = random_split(dataset, [train_size, test_size])
        self.system_dynamics = system_dynamics
        self.device = device
        self.optimizer = optimizer
        self.eval_traj_config = eval_traj_config
        self.plotter = plotter.Plotter()
        self.eval_traj_noise_scale = eval_traj_noise_scale
        self.system_dynamics_loss_weights = system_dynamics_loss_weights
        self.save_interval = save_interval
        self.max_iterations = max_iterations
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
        self.current_learning_iteration = 0

    def train(self):
        self.system_dynamics.train()
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + self.max_iterations
        num_batches = len(self.train_dataloader)
        for it in range(start_iter, tot_iter):
            self.log_dict = {}
            mean_system_state_loss = 0
            mean_system_sequence_loss = 0
            mean_system_bound_loss = 0
            mean_system_kl_loss = 0
            mean_system_extension_loss = 0
            mean_system_contact_loss = 0
            mean_system_termination_loss = 0
            for state, action, extension, contact, termination in self.train_dataloader:
                self.system_dynamics.reset()
                state, action = state.to(self.device), action.to(self.device)
                extension = extension.to(self.device) if extension.shape[-1] > 0 else None
                contact = contact.to(self.device) if contact.shape[-1] > 0 else None
                termination = termination.to(self.device) if termination.shape[-1] > 0 else None
                self.optimizer.zero_grad()
                state_loss, sequence_loss, bound_loss, kl_loss, extension_loss, contact_loss, termination_loss = self.system_dynamics.compute_loss(state, action, extension, contact, termination)
                loss = self.system_dynamics_loss_weights["state"] * state_loss \
                        + self.system_dynamics_loss_weights["sequence"] * sequence_loss \
                        + self.system_dynamics_loss_weights["bound"] * bound_loss \
                        + self.system_dynamics_loss_weights["kl"] * kl_loss \
                        + self.system_dynamics_loss_weights["extension"] * extension_loss \
                        + self.system_dynamics_loss_weights["contact"] * contact_loss \
                        + self.system_dynamics_loss_weights["termination"] * termination_loss
                loss.backward()
                self.optimizer.step()
                mean_system_state_loss += state_loss.item()
                mean_system_sequence_loss += sequence_loss.item()
                mean_system_bound_loss += bound_loss.item()
                mean_system_kl_loss += kl_loss.item()
                mean_system_extension_loss += extension_loss.item()
                mean_system_contact_loss += contact_loss.item()
                mean_system_termination_loss += termination_loss.item()
            mean_system_state_loss /= num_batches
            mean_system_sequence_loss /= num_batches
            mean_system_bound_loss /= num_batches
            mean_system_kl_loss /= num_batches
            mean_system_extension_loss /= num_batches
            mean_system_contact_loss /= num_batches
            mean_system_termination_loss /= num_batches
            print(f"[Model Training] Iteration {it}/{tot_iter} | Loss: {loss.item()}")
            self.current_learning_iteration = it
            self.log_dict.update({
                "Model/train_state_loss": mean_system_state_loss,
                "Model/train_sequence_loss": mean_system_sequence_loss,
                "Model/train_bound_loss": mean_system_bound_loss,
                "Model/train_kl_loss": mean_system_kl_loss,
                "Model/train_extension_loss": mean_system_extension_loss,
                "Model/train_contact_loss": mean_system_contact_loss,
                "Model/train_termination_loss": mean_system_termination_loss,
                })
            if it % self.save_interval == 0:
                self.evaluate()
                self.save(it)
            wandb.log(self.log_dict)
        self.evaluate()
        self.save(it)

    def _autoregressive_prediction(self, state_traj, action_traj, extension_traj, contact_traj, termination_traj):
        state_traj_pred = torch.zeros_like(state_traj, device=self.device)
        aleatoric_uncertainty_traj_pred = torch.zeros(state_traj.shape[0], state_traj.shape[1], device=self.device)
        epistemic_uncertainty_traj_pred = torch.zeros(state_traj.shape[0], state_traj.shape[1], device=self.device)
        action_traj_pred = action_traj.clone()
        extension_traj_pred = torch.zeros_like(extension_traj, device=self.device)
        contact_traj_pred = torch.zeros_like(contact_traj, device=self.device)
        termination_traj_pred = torch.zeros_like(termination_traj, device=self.device)
        
        state_traj_pred[:, :self.start_step] = state_traj[:, :self.start_step]
        extension_traj_pred[:, :self.start_step] = extension_traj[:, :self.start_step]
        contact_traj_pred[:, :self.start_step] = contact_traj[:, :self.start_step]
        termination_traj_pred[:, :self.start_step] = termination_traj[:, :self.start_step]

        self.system_dynamics.reset()
        with torch.inference_mode():
            for i in range(self.start_step, self.eval_traj_config["len_traj"]):
                if self.system_dynamics.architecture_config["type"] in ["rnn", "rssm"] and i > self.start_step:
                    state_input = state_traj_pred[:, i - 1:i]
                    action_input = action_traj_pred[:, i - 1:i]
                else:
                    state_input = state_traj_pred[:, i - self.start_step:i]
                    action_input = action_traj_pred[:, i - self.start_step:i]
                state_pred, aleatoric_uncertainty, epistemic_uncertainty, extension_pred, contact_pred, termination_pred = self.system_dynamics.forward(state_input, action_input)
                state_traj_pred[:, i] = state_pred
                aleatoric_uncertainty_traj_pred[:, i] = aleatoric_uncertainty
                epistemic_uncertainty_traj_pred[:, i] = epistemic_uncertainty
                if extension_pred is not None:
                    extension_traj_pred[:, i] = extension_pred
                if contact_pred is not None:
                    contact_traj_pred[:, i] = torch.sigmoid(contact_pred).round().int()
                if termination_pred is not None:
                    termination_traj_pred[:, i] = torch.sigmoid(termination_pred).round().int()
        return state_traj_pred, aleatoric_uncertainty_traj_pred, epistemic_uncertainty_traj_pred, action_traj_pred, extension_traj_pred, contact_traj_pred, termination_traj_pred
    
    def evaluate(self):
        self.system_dynamics.eval()
        num_batches = len(self.test_dataloader)
        mean_system_state_loss = 0
        mean_system_sequence_loss = 0
        mean_system_bound_loss = 0
        mean_system_kl_loss = 0
        mean_system_extension_loss = 0
        mean_system_contact_loss = 0
        mean_system_termination_loss = 0
        with torch.inference_mode():
            for state, action, extension, contact, termination in self.test_dataloader:
                self.system_dynamics.reset()
                state, action, extension, contact, termination = state.to(self.device), action.to(self.device), extension.to(self.device), contact.to(self.device), termination.to(self.device)
                state_loss, sequence_loss, bound_loss, kl_loss, extension_loss, contact_loss, termination_loss = self.system_dynamics.compute_loss(state, action, extension, contact, termination)
                mean_system_state_loss += state_loss.item()
                mean_system_sequence_loss += sequence_loss.item()
                mean_system_bound_loss += bound_loss.item()
                mean_system_kl_loss += kl_loss.item()
                mean_system_extension_loss += extension_loss.item()
                mean_system_contact_loss += contact_loss.item()
                mean_system_termination_loss += termination_loss.item()
        mean_system_state_loss /= num_batches
        mean_system_sequence_loss /= num_batches
        mean_system_bound_loss /= num_batches
        mean_system_kl_loss /= num_batches
        mean_system_extension_loss /= num_batches
        mean_system_contact_loss /= num_batches
        mean_system_termination_loss /= num_batches
        print(f"[Model Evaluation] Eval Loss: {mean_system_state_loss}")
        self.log_dict.update({
            "Model/eval_state_loss": mean_system_state_loss,
            "Model/eval_sequence_loss": mean_system_sequence_loss,
            "Model/eval_bound_loss": mean_system_bound_loss,
            "Model/eval_kl_loss": mean_system_kl_loss,
            "Model/eval_extension_loss": mean_system_extension_loss,
            "Model/eval_contact_loss": mean_system_contact_loss,
            "Model/eval_termination_loss": mean_system_termination_loss,
        })

        # evaluate and visualize trajectories
        if self.eval_traj_config is not None:
            fig, ax = plt.subplots(len(self.eval_traj_config["state_idx_dict"]) + 4, self.eval_traj_config["num_visualizations"], figsize=(10 * self.eval_traj_config["num_visualizations"], 10))
            state_traj, action_traj, extension_traj, contact_traj, termination_traj = self.eval_traj_config["traj_data"]
            self.plotter.plot_trajectories(
                ax,
                None,
                state_traj[:self.eval_traj_config["num_visualizations"]],
                action_traj[:self.eval_traj_config["num_visualizations"]],
                extension_traj[:self.eval_traj_config["num_visualizations"]],
                contact_traj[:self.eval_traj_config["num_visualizations"]],
                termination_traj[:self.eval_traj_config["num_visualizations"]],
                self.eval_traj_config["state_idx_dict"]
                )
            
            state_traj, action_traj = self.train_dataset.dataset.normalize(state_traj, action_traj)
            self.start_step = self.history_horizon
            state_traj_pred, _, _, action_traj_pred, extension_traj_pred, contact_traj_pred, termination_traj_pred = self._autoregressive_prediction(state_traj, action_traj, extension_traj, contact_traj, termination_traj)
            traj_autoregressive_error = ((state_traj_pred[:, self.start_step:] - state_traj[:, self.start_step:]).abs().sum(dim=-1) / state_traj[:, self.start_step:].abs().sum(dim=-1)).mean().item()
            print(f"[Model Evaluation] Autoregressive Error: {traj_autoregressive_error}")
            
            state_traj_pred, action_traj_pred = self.train_dataset.dataset.denormalize(state_traj_pred, action_traj_pred)
            self.plotter.plot_trajectories(
                ax,
                self.start_step,
                state_traj_pred[:self.eval_traj_config["num_visualizations"]],
                action_traj_pred[:self.eval_traj_config["num_visualizations"]],
                extension_traj_pred[:self.eval_traj_config["num_visualizations"]],
                contact_traj_pred[:self.eval_traj_config["num_visualizations"]],
                termination_traj_pred[:self.eval_traj_config["num_visualizations"]],
                self.eval_traj_config["state_idx_dict"],
                prediction=True
                )
            self.log_dict.update({
                "Model/traj_autoregressive_error": traj_autoregressive_error,
                "Model/visualize_trajs": wandb.Image(fig) if self.eval_traj_config is not None else None,
                })
            
            for noise_scale in self.eval_traj_noise_scale:
                state_traj_noised = state_traj + torch.randn_like(state_traj) * noise_scale
                action_traj_noised = action_traj + torch.randn_like(action_traj) * noise_scale
                state_traj_pred_noised, _, _, _, _, _, _ = self._autoregressive_prediction(state_traj_noised, action_traj_noised, extension_traj, contact_traj, termination_traj)
                traj_autoregressive_error_noised = ((state_traj_pred_noised[:, self.start_step:] - state_traj_noised[:, self.start_step:]).abs().sum(dim=-1) / state_traj_noised[:, self.start_step:].abs().sum(dim=-1)).mean().item()
                print(f"[Model Evaluation] Noised Autoregressive Error with Level {noise_scale}: {traj_autoregressive_error_noised}")
                self.log_dict.update({
                    f"Model/traj_autoregressive_error_noised_{noise_scale}": traj_autoregressive_error_noised,
                    })
            plt.close(fig)
        self.system_dynamics.train()

        
    def save(self, it):
        torch.save(
            {
                "system_dynamics_state_dict": self.system_dynamics.state_dict(),
                "iter": it,
                },
            self.log_dir + f"/model_{it}.pt"
            )