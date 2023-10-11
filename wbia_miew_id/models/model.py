import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from losses import ECELoss
from .heads import ArcMarginProduct, ElasticArcFace, ArcFaceSubCenterDynamic

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
        
    def __repr__(self):
        return self.__class__.__name__ + \
                '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + \
                ', ' + 'eps=' + str(self.eps) + ')'
    

            
class MiewIdNet(nn.Module):

    def __init__(self,
                 n_classes,
                 model_name='efficientnet_b0',
                 use_fc=False,
                 fc_dim=512,
                 dropout=0.0,
                 loss_module='softmax',
                 s=30.0,
                 margin=0.50,
                 ls_eps=0.0,
                 theta_zero=0.785,
                 pretrained=True,
                 margins=None,
                 k=None):
        """
        """
        super(MiewIdNet, self).__init__()
        print('Building Model Backbone for {} model'.format(model_name))

        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        final_in_features = self.backbone.classifier.in_features
        
        self.backbone.classifier = nn.Identity()
        self.backbone.global_pool = nn.Identity()
        
        self.pooling =  GeM()
        self.bn = nn.BatchNorm1d(final_in_features)
        self.use_fc = use_fc
        if use_fc:
            self.dropout = nn.Dropout(p=dropout)
            self.bn = nn.BatchNorm1d(fc_dim)
            self.bn.bias.requires_grad_(False)
            self.fc = nn.Linear(final_in_features, n_classes, bias = False)            
            self.bn.apply(weights_init_kaiming)
            self.fc.apply(weights_init_classifier)
            final_in_features = fc_dim

        self.loss_module = loss_module
        if loss_module == 'arcface':
            self.final = ElasticArcFace(final_in_features, n_classes,
                                          s=s, m=margin)
        elif loss_module == 'arcface_subcenter_dynamic':
            if margins is None:
                margins = [0.3] * n_classes
            self.final = ArcFaceSubCenterDynamic(
                embedding_dim=final_in_features, 
                output_classes=n_classes, 
                margins=margins,
                s=s,
                k=k )
        # elif loss_module == 'cosface':
        #     self.final = AddMarginProduct(final_in_features, n_classes, s=s, m=margin)
        # elif loss_module == 'adacos':
        #     self.final = AdaCos(final_in_features, n_classes, m=margin, theta_zero=theta_zero)
        else:
            self.final = nn.Linear(final_in_features, n_classes)

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x, label=None):
        feature = self.extract_feat(x)
        if not self.training:
            return feature
        else:
            assert label is not None
        if self.loss_module in ('arcface', 'arcface_subcenter_dynamic'):
            logits = self.final(feature, label)
        else:
            logits = self.final(feature)
        return logits

    def extract_feat(self, x):
        batch_size = x.shape[0]
        x = self.backbone(x)
        x = self.pooling(x).view(batch_size, -1)
        x = self.bn(x)
        if self.use_fc:
            x1 = self.dropout(x)
            x1 = self.bn(x1)
            x1 = self.fc(x1)
            

        return x

class MiewIdNetTS(MiewIdNet):

    def __init__(self,
                 temperature=1.0,
                 **kwargs):
        """
        """
        super().__init__(**kwargs)

        self.temperature = nn.Parameter(torch.ones(1) * temperature)

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of the logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature
    
    def set_temperature(self, valid_loader, device):
        """
        Tune the temperature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        nll_criterion = nn.CrossEntropyLoss().to(device)
        ece_criterion = ECELoss().to(device)

        # First: collect all the logits and labels for the validation set
        print("Collecting logits and labels in validation set")
        logits_list = []
        labels_list = []
        from tqdm import tqdm
        with torch.no_grad():
            for batch in tqdm(valid_loader):
                input = batch['image'].to(device)
                label = batch['label'].to(device)
                logits = self(input)
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list).to(device)
            labels = torch.cat(labels_list).to(device)

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        print("Optimizing temperature w.r.t. NLL")
        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        print("Calculating post-temperature scaling NLL and ECE")
        self.after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        self.after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        print('Optimal temperature: %.3f' % self.temperature.item())
        print('After temperature - NLL: %.3f, ECE: %.3f' % (self.after_temperature_nll, self.after_temperature_ece))

        return self

    def forward(self, x, label=None):
        feature = self.extract_feat(x)
        if not self.training:
            return feature
        else:
            assert label is not None
        if self.loss_module in ('arcface', 'arcface_subcenter_dynamic'):
            logits = self.final(feature, label)
        else:
            logits = self.final(feature)

        logits = self.temperature_scale(logits)

        return logits

    def extract_feat(self, x):
        batch_size = x.shape[0]
        x = self.backbone(x)
        x = self.pooling(x).view(batch_size, -1)
        x = self.bn(x)
        if self.use_fc:
            x1 = self.dropout(x)
            x1 = self.bn(x1)
            x1 = self.fc(x1)
    
        return x