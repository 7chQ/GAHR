import random
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from misc import utils
from misc.utils import is_using_distributed
from text_utils.tokenizer import tokenize
from .visual_transformer import visual_transformer
from .text_transformer import text_transformers
from .eda import EDA
from .shared_modules import AllGather
from .bi_crossattention import BiCrossAttention
from einops import rearrange
from model import lorentz as L


class CLIP(nn.Module):
    def __init__(self, config, image_encode, text_encode, num_classes=11003, eps=1e-2):
        super().__init__()
        self.visual = image_encode
        self.encode_text = text_encode
        self.embed_dim = config.model.embed_dim
        self.lamda = config.experiment.lamda

        self.use_gather = config.model.use_gather
        self.logit_scale = nn.Parameter(torch.ones([]))
        nn.init.constant_(self.logit_scale, np.log(1 / 0.07))
        self.config = config
        self.eda = EDA()
        self.eps = eps

        self.Bi_cross_attention =  BiCrossAttention(config.model.embed_dim)
        self.predictor = nn.Sequential(nn.Linear(config.model.embed_dim, config.model.embed_dim // 2),
                                        nn.ReLU(inplace=True), 
                                        nn.Linear(config.model.embed_dim // 2, config.model.embed_dim)) 
        self.local_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.1))

        self.temp = nn.Parameter(torch.ones([]) * 0.07)   
        self.curv = nn.Parameter(
            torch.tensor(1.0).log(), requires_grad=True
        )

    def bi2t(self, embed_A, embed_B, norm=True):
        
            logit_scale = self.local_logit_scale.exp()
            embed_A = rearrange(embed_A, "b n1 n2 -> (b n1) n2")
            embed_B = rearrange(embed_B, "b n1 n2 -> (b n1) n2")
            if norm:
                embed_A = F.normalize(embed_A, dim=-1, p=2)
                embed_B = F.normalize(embed_B, dim=-1, p=2)
            self.lc_labels = torch.arange(embed_B.size(0), device=embed_B.device).long()
            logits_per_image = logit_scale * embed_B @ embed_A.t()
            logits_per_text = logit_scale * embed_A @ embed_B.t()
            image_loss = F.cross_entropy(logits_per_image, self.lc_labels)
            text_loss = F.cross_entropy(logits_per_text, self.lc_labels)
            loss = (image_loss + text_loss) / 2   
            return loss 
        
    def bt2i(self, x, y, predictor):
            p_x = predictor(x)
            p_y = predictor(y)
            z_x = x.detach()
            z_y = y.detach() 
            return - (F.cosine_similarity(p_x, z_y, dim=-1).mean() + F.cosine_similarity(p_y, z_x, dim=-1).mean()) * 0.5   

    def forward(self, input, alpha):
        ret = dict()

        images = input['image'].to(self.config.device)
        texts = input['caption']
        texts_bt = input['caption_bt']

        # back translation
        if self.config.experiment.back_trans:
            for i in range(len(texts)):
                if random.random() < self.config.experiment.backtrans_p:
                    texts[i] = texts_bt[i]

        # random deletion
        cap_new = []
        for text in texts:
            eda_alpha = self.config.experiment.eda_alpha
            cap_new.append(self.eda.random_deletion(text, eda_alpha))
        texts = cap_new

        text_tokens = tokenize(texts, context_length=self.config.experiment.text_length).to(self.config.device)
        ids = input['id'].to(self.config.device)

        image_features, image_seq_embeddings = self.encode_image(images, self.curv, return_dense=True)
        text_features, text_seq_embeddings = self.encode_text(text_tokens, self.curv, return_dense=True)
        image_features_norm = image_features
        text_features_norm = text_features
        image_features_norm_gathered = self.all_gather(image_features_norm)
        text_features_norm_gathered = self.all_gather(text_features_norm)

        logit_scale = self.logit_scale.exp()
        logit_scale.data = torch.clamp(logit_scale.data, max=100)

        idx = ids.view(-1, 1)
        gathered_ids = self.all_gather(ids)
        idx_all = gathered_ids.view(1, -1)
        pos_idx = torch.eq(idx, idx_all).float()
        sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)

        with torch.no_grad():
            image_features_s, image_seq_embeddings_s = self.encode_image(images, self.curv, return_dense=True)
            text_features_s, text_seq_embeddings_s = self.encode_text(text_tokens, self.curv, return_dense=True)
            image_features_s_norm = image_features_s
            text_features_s_norm = text_features_s
            image_features_s_norm_gathered = self.all_gather(image_features_s_norm)
            text_features_s_norm_gathered = self.all_gather(text_features_s_norm)
        nitc_loss = self.sms_contrastive(image_features_norm, text_features_norm, image_features_s_norm,
                                          text_features_s_norm,
                                          image_features_norm_gathered, text_features_norm_gathered,
                                          image_features_s_norm_gathered, text_features_s_norm_gathered,
                                          sim_targets, alpha, logit_scale, self.curv)

        ret['nitc_loss'] = nitc_loss * self.config.experiment.nitc_ratio

        if self.config.experiment.citc:
            logits_image_per_image = logit_scale * (L.pairwise_inner(image_features_norm_gathered, image_features_norm_gathered, self.curv.exp()))
            logits_text_per_text = logit_scale * (L.pairwise_inner(text_features_norm_gathered, text_features_norm_gathered, self.curv.exp()))
            inmodal_cyclic_loss = (logits_image_per_image - logits_text_per_text).square().mean() / (
                    logit_scale * logit_scale)
            logits_text_per_image = logit_scale * image_features_norm_gathered @ text_features_norm_gathered.t()
            logits_image_per_text = logit_scale * text_features_norm_gathered @ image_features_norm_gathered.t()
            crossmodal_cyclic_loss = (logits_text_per_image - logits_image_per_text).square().mean() / (
                    logit_scale * logit_scale)
            citc_loss = self.config.experiment.citc_lambda1 * inmodal_cyclic_loss + self.config.experiment.citc_lambda2 * crossmodal_cyclic_loss
            ret['citc_loss'] = citc_loss * self.config.experiment.citc_ratio

        if self.config.experiment.ritc:
            logits_per_image_1 = logit_scale * (-L.pairwise_dist(image_features_norm, text_features_norm_gathered, self.curv.exp()))
            logits_per_text_1 = logit_scale * (-L.pairwise_dist(text_features_norm, image_features_norm_gathered, self.curv.exp()))
            img_log = F.log_softmax(logits_per_image_1, dim=1)
            txt_log = F.log_softmax(logits_per_text_1, dim=1)
            target_log = (sim_targets + self.eps).log()
            kl_img = F.kl_div(target_log, img_log, log_target=True, reduction='batchmean')
            kl_txt = F.kl_div(target_log, txt_log, log_target=True, reduction='batchmean')
            ritc_loss = 0.5 * (kl_img + kl_txt)
            ret['ritc_loss'] = ritc_loss * self.config.experiment.ritc_ratio
        
        if self.config.experiment.bai:
            text_to_local_image_embed, text_to_local_image_atten, image_to_local_text_embed, image_to_local_text_atten  \
                = self.Bi_cross_attention(image_seq_embeddings, text_seq_embeddings) 
            image_loss = self.bt2i(image_seq_embeddings, text_to_local_image_embed, self.predictor)
            text_loss = self.bi2t(text_seq_embeddings, image_to_local_text_embed)
            ret['bai_loss'] =  (image_loss + text_loss) * self.config.experiment.bai_ratio 
            
        return ret
    
    def sms_contrastive(self, image_features, text_features, image_features_s, text_features_s,
                         image_features_gathered, text_features_gathered, image_features_s_gathered,
                         text_features_s_gathered,
                         sim_targets, alpha, logit_scale, curv):
        _curv = curv.exp()
        with torch.autocast("cuda", dtype=torch.float32):
            with torch.no_grad():
                sim_i2t_s = logit_scale * (-L.pairwise_dist(image_features_s, text_features_s_gathered, _curv))
                sim_t2i_s = logit_scale * (-L.pairwise_dist(text_features_s, image_features_s_gathered, _curv))
                sim_i2i_s = -L.pairwise_dist(image_features_s, text_features_s_gathered, _curv)
                sim_t2t_s = -L.pairwise_dist(text_features_s, image_features_s_gathered, _curv)
                sim_i2t_targets = alpha * F.softmax(sim_i2t_s, dim=1) + (1 - alpha) * sim_targets
                sim_t2i_targets = alpha * F.softmax(sim_t2i_s, dim=1) + (1 - alpha) * sim_targets  # soft + hard
            sim_i2t = logit_scale * (-L.pairwise_dist(image_features, text_features_gathered, _curv))
            sim_t2i = logit_scale * (-L.pairwise_dist(text_features, image_features_gathered, _curv))
            loss_i2t = -torch.sum(F.log_softmax(sim_i2t + self.lamda * logit_scale * (torch.diag(sim_i2i_s).unsqueeze(1) - sim_i2i_s), dim=1) * sim_i2t_targets, dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(sim_t2i + self.lamda * logit_scale * (torch.diag(sim_t2t_s).unsqueeze(1) - sim_t2t_s), dim=1) * sim_t2i_targets, dim=1).mean()
            loss_ita = (loss_i2t + loss_t2i) / 2        
        return loss_ita     


    @property
    def dtype(self):
        try:
            return self.visual.conv1.weight.dtype
        except:
            try:
                return self.visual.head.weight.dtype
            except:
                try:
                    return self.visual.stem[0].weight.dtype
                except:
                    return self.encode_text.text_projection.weight.dtype

    def encode_image(self, image, curv, return_dense=False):
        if return_dense:
            output = self.visual(image.type(self.dtype), curv, return_dense=return_dense)
            return output
        output = self.visual(image.type(self.dtype), curv)
        return output

    def all_gather(self, input):
        if not self.use_gather or not is_using_distributed():
            return input
        output = AllGather.apply(input)
        output = output.view(-1, *(output.shape[2:]))
        return output


def clip_vitb(config, num_classes=11003):
    image_encode = visual_transformer(config)
    text_encode = text_transformers(config)
    model = CLIP(config, image_encode, text_encode, num_classes, config.experiment.ritc_eps)
    return model
