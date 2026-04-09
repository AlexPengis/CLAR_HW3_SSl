import torch.nn as nn
from resnet.resnet_1D import *
from resnet.resnet_2D import *
from resnet.ast_2D import CreateAST2D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    r"""
        CLAR model
            input audio and spectogram augmented
            act as input to the encoders made by resnets
            finnaly the outputs are resized by projection heads to 128 vectors
        Returns:
            - Embedding of audio and spectograms
            - Output of resnet1d and resnet2d to traing the evaluationhead
    """

    def __init__(self,img_channels = 3, num_classes = 35, unsupervised = False):
        super(Net, self).__init__()
        self.unsupervised = unsupervised
        ####################### ENCODER ###################################
        self.resnet_1D = CreateResNet1D(num_classes=num_classes).to(device)
        self.resnet_2D = CreateResNet2D(img_channels=img_channels,num_classes=num_classes).to(device)
        
        ####################### PROJECTION HEAD ###########################
        self.projectionHead = nn.Sequential(
                                    nn.Linear(512, 256),
                                    nn.BatchNorm1d(256),
                                    nn.ReLU(),
                                    nn.Linear(256, 128) # Output goes to the contrastive loss!         
                                    )
        
       # Last layer of the projection head used to semi-supervised Categorial cross Entropy
        if unsupervised == True:
            self.output = nn.Sequential(
                                    nn.Linear(512, 256),
                                    nn.BatchNorm1d(256),
                                    nn.ReLU(),
                                    nn.Linear(256, 256),         
                                    nn.BatchNorm1d(256),
                                    nn.ReLU(),
                                    nn.Linear(256, 128) # Output goes to the contrastive loss!         
                                    )
        else:
            
            self.output = nn.Sequential(
                                    nn.BatchNorm1d(128),
                                    nn.ReLU(),
                                    nn.Linear(128, num_classes) # Output goes to the cross Entropy         
                                    )
        
        ####################################################################
        
    def forward(self, input_spectogram, input_audio):
        """
            resnet2d and resnet1d output is [BS, 512, 1, 1]
            Output:
                - audio_emb, specs_emb used for contrastive loss
                - audio, spectograms used for Evaluation layer
                - output used for semi supervised - cross entropy
        """
        
        audio = self.resnet_1D(input_audio)
        audio = audio.squeeze()
        spectograms = self.resnet_2D(input_spectogram)
        spectograms = spectograms.squeeze()

        # If unsupervised the last layer of projection head has 128 output dimension 
        if self.unsupervised == True:
            audio_emb = self.output(audio)
            specs_emb = self.output(spectograms)   

            return audio_emb, specs_emb, audio, spectograms
        ##################################################

        audio_emb = self.projectionHead(audio)
        specs_emb = self.projectionHead(spectograms)
        
        output = self.output(torch.cat([audio_emb, specs_emb], dim=0))
                
        # should be 128 size
        return audio_emb, specs_emb, audio, spectograms, output


class Net2(nn.Module):
    r"""
        CLAR model — AST variant
            Same dual-encoder structure as Net but replaces the ResNet18-2D
            spectrogram encoder with an Audio Spectrogram Transformer (AST-lite).

            AST splits the mel spectrogram into 16×16 patches, encodes them
            with a 4-layer Transformer, and uses the [CLS] token as the
            spectrogram embedding.  The raw-audio branch keeps ResNet18-1D.

        Returns:
            Same tuple as Net for full drop-in compatibility with all
            training scripts.
    """

    def __init__(self, img_channels=3, num_classes=10, unsupervised=False):
        super(Net2, self).__init__()
        self.unsupervised = unsupervised

        ####################### ENCODERS ##################################
        self.resnet_1D = CreateResNet1D(num_classes=num_classes).to(device)
        self.ast_2D    = CreateAST2D(img_channels=img_channels,
                                     num_classes=num_classes).to(device)

        ####################### PROJECTION HEAD ###########################
        self.projectionHead = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

        if unsupervised:
            self.output = nn.Sequential(
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 128),
            )
        else:
            self.output = nn.Sequential(
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, num_classes),
            )

    def forward(self, input_spectogram, input_audio):
        """
        Args:
            input_spectogram : [B, 3, 128, 126]
            input_audio      : [B, 1, 16000]
        Returns:
            Same tuple as Net.
        """
        audio      = self.resnet_1D(input_audio).squeeze()
        spectograms = self.ast_2D(input_spectogram).squeeze()

        if self.unsupervised:
            audio_emb  = self.output(audio)
            specs_emb  = self.output(spectograms)
            return audio_emb, specs_emb, audio, spectograms

        audio_emb  = self.projectionHead(audio)
        specs_emb  = self.projectionHead(spectograms)
        output     = self.output(torch.cat([audio_emb, specs_emb], dim=0))

        return audio_emb, specs_emb, audio, spectograms, output

