import torch.nn as nn

class Classifier(nn.Module):  # classifies NPI outputs
    def __init__(self, n=200, m=768, k=1):
        """
        input_activs_shape: tuple of (b, n, m, 1)
            b is the number of batches
            n x m x 1 slices contain the elements of the original activations, flattened into a 2D array
        target_label: tuple of (b, 1, m, 1)
            the desired label for the predicted activations, as passed into the NPI network
        """
        super(Classifier, self).__init__()

        print("Classifier INIT", flush=True)
        self.n = n
        self.m = m
        self.k = k
        self.N = self.n * self.m

        fact1 = 2 ** 4
        fact2 = 2 ** 5
        fact3 = 2 ** 6

        print("Defining classifier model", flush=True)

        self.model = nn.Sequential(
            nn.Linear(self.n * self.m * self.k, self.n // fact1),
            nn.ReLU(),
            nn.Linear(self.n // fact1, self.n // fact2),
            nn.ReLU(),
            nn.Linear(self.n // fact2, self.n // fact3),
            nn.ReLU(),
            nn.Linear(self.n // fact3, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x.view(-1, self.n * self.m * self.k))


class GenerationClassifier(nn.Module):  # classifies NPI outputs
    def __init__(self, input_activs_shape, input_targ_shape):
        """
        input_activs_shape: tuple of (n, m, 1)
            n x m x 1 slices contain the elements of the original activations, flattened into a 2D array
        target_label: tuple of (b, 1, m, 1)
            the desired label for the predicted activations, as passed into the NPI network
        """
        super(GenerationClassifier, self).__init__()

        print("GenerationClassifier INIT")
        self.n = input_activs_shape[0]
        self.m = input_activs_shape[1]
        self.k = input_activs_shape[2]

        self.l = 1

        fact1 = 2 ** 3
        fact2 = 2 ** 4
        fact3 = 2 ** 5

        print("Defining GenerationClassifier model")

        self.layer1 = nn.Sequential(nn.Linear(self.n * self.m * self.k, self.n // fact1),
                                    nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(self.n // fact1, self.n // fact1),
                                    nn.ReLU())
        self.layer3 = nn.Sequential(nn.Linear(self.n // fact1, self.n // fact2),
                                    nn.ReLU())
        self.layer4 = nn.Sequential(nn.Linear(self.n // fact2, self.n // fact2),
                                    nn.ReLU())
        self.layer5 = nn.Sequential(nn.Linear(self.n // fact2, self.n // fact3),
                                    nn.ReLU())
        self.layer6 = nn.Sequential(nn.Linear(self.n // fact3, self.n // fact3),
                                    nn.ReLU())
        self.layer7 = nn.Sequential(nn.Linear(self.n // fact3, self.l * self.k),
                                    nn.Sigmoid())

    def forward(self, x):
        metadata = {'ordered_hidden_activations': [], 'final_out_preview': None, 'final_out_returned': None}

        out1 = self.layer1(x.view(-1, self.n * self.m * self.k))
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        final_out = self.layer7(out6)

        # metadata['ordered_hidden_activations'] = [out1.detach().data.cpu().numpy(),
        #                                          out2.detach().data.cpu().numpy(), 
        #                                          out3.detach().data.cpu().numpy(), 
        #                                          out4.detach().data.cpu().numpy(), 
        #                                          out5.detach().data.cpu().numpy(), 
        #                                          out6.detach().data.cpu().numpy(), 
        #                                          ]
        # metadata['final_out_preview'] = final_out.detach().data.cpu().numpy()
        # metadata['final_out_returned'] = final_out.view(-1, 1, self.l, self.k).detach().data.cpu().numpy()
        return final_out.view(-1, 1, self.l, self.k)  # , metadata

class ContentClassifier(nn.Module):  # classifies NPI outputs
    def __init__(self, input_activs_shape, input_targ_shape):
        raise NotImplementedError("Content classifier should be pre-trained")
        """
        input_activs_shape: tuple of (b, n, m, 1)
            b is the number of batches
            n x m x 1 slices contain the elements of the original activations, flattened into a 2D array
        """
        super(ContentClassifier, self).__init__()

        print("ContentClassifier INIT")
        self.b = input_activs_shape[0]
        self.n = input_activs_shape[1]
        self.m = input_activs_shape[2]
        self.k = input_activs_shape[3]

        self.l = 1  # input_targ_shape[2]

        fact1 = 2 ** 3
        fact2 = 2 ** 3
        fact3 = 2 ** 3

        print("Defining ContentClassifier model")
        self.linear1 = nn.Sequential(nn.Linear(self.n * self.m * self.k, self.n // fact1),
                                     nn.ReLU())
        self.linear1Post = nn.Sequential(nn.Linear(self.n // fact1, self.n // fact1),
                                         nn.ReLU())
        self.linear2 = nn.Sequential(nn.Linear(self.n // fact1, self.n // fact1),
                                     nn.ReLU())
        self.linear3 = nn.Sequential(nn.Linear(self.n // fact1, self.n // fact2),
                                     nn.ReLU())
        self.linear4 = nn.Sequential(nn.Linear(self.n // fact2, self.n // fact2),
                                     nn.ReLU())
        self.linear5 = nn.Sequential(nn.Linear(self.n // fact2, self.n // fact3),
                                     nn.ReLU())
        self.linear6 = nn.Sequential(nn.Linear(self.n // fact3, self.n // fact3),
                                     nn.ReLU())
        self.linear7Pre = nn.Sequential(nn.Linear(self.n // fact3, self.n // fact3),
                                        nn.ReLU())
        self.linear7 = nn.Sequential(nn.Linear(self.n // fact3, 1 * self.l * self.k),
                                     nn.Sigmoid())

    def forward(self, x):
        metadata = {'ordered_hidden_activations': [], 'final_out_preview': None, 'final_out_returned': None}
        out1 = self.linear1(x.view(-1, self.n * self.m * self.k))
        out1Post = self.linear1Post(out1)
        out2 = self.linear2(out1Post)
        out3 = self.linear3(out2)
        out4 = self.linear4(out3)
        out5 = self.linear5(out4)
        out6 = self.linear6(out5)
        out7Pre = self.linear7Pre(out6)
        final_out = self.linear7(out6)

        metadata['ordered_hidden_activations'] = [out1.detach().data.cpu().numpy(),
                                                  out1Post.detach().data.cpu().numpy(),
                                                  out2.detach().data.cpu().numpy(),
                                                  out3.detach().data.cpu().numpy(),
                                                  out4.detach().data.cpu().numpy(),
                                                  out5.detach().data.cpu().numpy(),
                                                  out6.detach().data.cpu().numpy(),
                                                  out7Pre.detach().data.cpu().numpy(),
                                                  ]
        metadata['final_out_preview'] = final_out.detach().data.cpu().numpy()
        metadata['final_out_returned'] = final_out.view(-1, 1, self.l, self.k).detach().data.cpu().numpy()
        return final_out.view(-1, 1, self.l, self.k), metadata
