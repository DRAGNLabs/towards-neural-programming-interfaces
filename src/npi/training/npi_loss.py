import torch
import torch.nn as nn

class NPILoss(nn.Module):
    def __init__(
        self,
        discrim_coeff,
        style_coeff,
        similarity_coeff,
        loss_boosting_coeff,
        style_class_model=None,
        discrim_model=None,
    ):
        super(NPILoss, self).__init__()
        self.gamma = discrim_coeff
        self.alpha = style_coeff
        self.beta = similarity_coeff
        self.loss_boosting_coeff = loss_boosting_coeff
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()

        if discrim_model is not None:
            self.discrim_model = discrim_model
        if style_class_model is not None:
            self.style_class_model = style_class_model
        pass

    def forward(
        self,
        predicted_activs,
        true_activs,
        target_label,
        return_loss_data=False,
    ):
        """
        predicted_activs: torch tensor of shape (n, m, 1, b)
            b is the number of batches
            n x m x 1 slices contain the elements of the predicted activations, flattened into a 2D array

        true_activs: torch tensor of shape (n, m, 1, b)
            b is the number of batches
            n x m x 1 slices contain the elements of the original activations, flattened into a 2D array

        target_label: torch tensor of shape (1, m, 1, b)
            the desired label for the predicted activations, as passed into the NPI network

        classifier_model: an updated classifier model (optional: use for adversarial training)
        """
        generation_classifier_labels, _ = self.discrim_model(
            predicted_activs
        )
        content_classifier_labels = (
            self.style_class_model(predicted_activs).unsqueeze(1).unsqueeze(3)
        )
        aggregate_size = torch.cat(
            (generation_classifier_labels, content_classifier_labels), dim=2
        ).size()
        classifier_labels = torch.zeros(aggregate_size, dtype=torch.float64).cuda()
        classifier_labels[:, :, 0, :] = generation_classifier_labels[:, :, 0, :]
        classifier_labels[:, :, 1, :] = content_classifier_labels[
            :, :, 0, :
        ]  # 1: to 1 and to 0

        new_discrim_score = self.gamma * self.bce(
            classifier_labels[:, :, 0, :], target_label[:, :, 0, :].double()
        )
        new_style_score = self.alpha * self.bce(
            classifier_labels[:, :, 1, :], target_label[:, :, 1, :].double()
        )  # 1: to 1
        old_content_score = self.beta * self.mse(predicted_activs, true_activs)

        if return_loss_data:
            return self.loss_boosting_coeff * (
                new_discrim_score + new_style_score + old_content_score
            ), {
                "gen_class_loss": new_discrim_score.item(),
                "content_class_loss": new_style_score.item(),
                "similarity_loss": old_content_score.item(),
            }
        return self.loss_boosting_coeff * (
            new_discrim_score + new_style_score + old_content_score
        )
