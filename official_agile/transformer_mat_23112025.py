#!/usr/bin/env python
"""
Molecule Attention Transformer (MAT) for AGILE Platform
State-of-the-art architecture for molecular property prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool
import math

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism for molecular graphs"""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections and reshape for multi-head attention
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        
        # Reshape and apply output projection
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(context)
        
        # Residual connection and layer normalization
        output = self.layer_norm(query + self.dropout(output))
        
        return output, attn_weights


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with feed-forward network"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention
        attn_output, attn_weights = self.self_attn(x, x, x, mask)
        
        # Feed-forward network
        ffn_output = self.ffn(attn_output)
        output = self.layer_norm(attn_output + self.dropout(ffn_output))
        
        return output, attn_weights


class MoleculeAttentionTransformer(nn.Module):
    """
    State-of-the-art Molecule Attention Transformer for AGILE
    
    Key improvements over GIN:
    1. Self-attention captures long-range dependencies
    2. Multi-head attention learns diverse structural patterns
    3. Positional encoding preserves molecular geometry
    4. Auxiliary tasks improve representation learning
    """
    
    def __init__(
        self,
        num_atom_types=119,
        num_bond_types=5,
        d_model=512,
        num_heads=8,
        num_layers=6,
        d_ff=2048,
        dropout=0.1,
        num_descriptors=813,
        use_auxiliary=True
    ):
        super().__init__()
        
        # Atom and bond embeddings
        self.atom_embedding = nn.Embedding(num_atom_types, d_model)
        self.bond_embedding = nn.Embedding(num_bond_types, d_model)
        
        # Positional encoding for molecular structure
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder layers
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Descriptor encoder (same as original AGILE)
        self.descriptor_encoder = nn.Sequential(
            nn.Linear(num_descriptors, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU()
        )
        
        # Multi-modal fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(d_model + 256, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Main prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
        
        # Auxiliary task heads (improve representation learning)
        self.use_auxiliary = use_auxiliary
        if use_auxiliary:
            # Predict molecular weight
            self.mw_head = nn.Linear(d_model, 1)
            
            # Predict number of rotatable bonds
            self.rot_bonds_head = nn.Linear(d_model, 1)
            
            # Predict lipophilicity (LogP)
            self.logp_head = nn.Linear(d_model, 1)
            
            # Contrastive learning head
            self.contrastive_head = nn.Sequential(
                nn.Linear(d_model, 256),
                nn.GELU(),
                nn.Linear(256, 128)
            )
    
    def forward(self, batch, return_attention=False):
        # Get atom features
        x = self.atom_embedding(batch.x[:, 0])
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Store attention weights if requested
        attention_weights = []
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x, attn = layer(x, mask=None)
            if return_attention:
                attention_weights.append(attn)
        
        # Global pooling (mean + max)
        graph_repr_mean = global_mean_pool(x, batch.batch)
        graph_repr_max = global_max_pool(x, batch.batch)
        graph_repr = (graph_repr_mean + graph_repr_max) / 2
        
        # Encode molecular descriptors
        desc_repr = self.descriptor_encoder(batch.features)
        
        # Fuse representations
        combined = torch.cat([graph_repr, desc_repr], dim=1)
        fused = self.fusion_layer(combined)
        
        # Main prediction
        prediction = self.prediction_head(fused)
        
        # Auxiliary predictions
        auxiliary_outputs = {}
        if self.use_auxiliary and self.training:
            auxiliary_outputs['mw'] = self.mw_head(graph_repr)
            auxiliary_outputs['rot_bonds'] = self.rot_bonds_head(graph_repr)
            auxiliary_outputs['logp'] = self.logp_head(graph_repr)
            auxiliary_outputs['contrastive'] = self.contrastive_head(graph_repr)
        
        if return_attention:
            return prediction, auxiliary_outputs, attention_weights
        
        return prediction, auxiliary_outputs


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss with uncertainty weighting
    Automatically balances task weights during training
    """
    
    def __init__(self, num_tasks=4):
        super().__init__()
        # Learnable task uncertainty parameters
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
    
    def forward(self, predictions, targets, auxiliary_preds, auxiliary_targets):
        # Main task loss (mTP prediction)
        main_loss = F.mse_loss(predictions, targets)
        
        # Auxiliary task losses
        mw_loss = F.mse_loss(auxiliary_preds['mw'], auxiliary_targets['mw'])
        rot_loss = F.mse_loss(auxiliary_preds['rot_bonds'], auxiliary_targets['rot_bonds'])
        logp_loss = F.mse_loss(auxiliary_preds['logp'], auxiliary_targets['logp'])
        
        # Uncertainty-weighted multi-task loss
        precision_main = torch.exp(-self.log_vars[0])
        precision_mw = torch.exp(-self.log_vars[1])
        precision_rot = torch.exp(-self.log_vars[2])
        precision_logp = torch.exp(-self.log_vars[3])
        
        total_loss = (
            precision_main * main_loss + self.log_vars[0] +
            0.3 * (precision_mw * mw_loss + self.log_vars[1]) +
            0.3 * (precision_rot * rot_loss + self.log_vars[2]) +
            0.3 * (precision_logp * logp_loss + self.log_vars[3])
        )
        
        return total_loss, {
            'main': main_loss.item(),
            'mw': mw_loss.item(),
            'rot': rot_loss.item(),
            'logp': logp_loss.item()
        }


def train_mat_model(model, train_loader, val_loader, epochs=100, lr=0.0001):
    """
    Training loop with advanced techniques:
    - Warmup learning rate schedule
    - Gradient clipping
    - Mixed precision training
    - Early stopping with patience
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Multi-task loss
    criterion = MultiTaskLoss(num_tasks=4).to(device)
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=lr,
        weight_decay=1e-5,
        betas=(0.9, 0.999)
    )
    
    # Cosine annealing with warm restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-7
    )
    
    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler()
    
    best_val_corr = -1.0
    patience = 20
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                pred, aux_pred = model(batch)
                
                # Compute auxiliary targets from descriptors
                aux_targets = {
                    'mw': batch.descriptors[:, 0].unsqueeze(1),  # MW descriptor
                    'rot_bonds': batch.descriptors[:, 5].unsqueeze(1),
                    'logp': batch.descriptors[:, 10].unsqueeze(1)
                }
                
                loss, loss_dict = criterion(pred, batch.y, aux_pred, aux_targets)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred, _ = model(batch)
                val_preds.extend(pred.cpu().numpy())
                val_targets.extend(batch.y.cpu().numpy())
        
        # Calculate metrics
        from scipy.stats import pearsonr
        val_corr = pearsonr(val_preds, val_targets)[0]
        val_rmse = torch.sqrt(torch.tensor(
            [(p - t)**2 for p, t in zip(val_preds, val_targets)]
        ).mean())
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"  Val Correlation: {val_corr:.4f}")
        print(f"  Val RMSE: {val_rmse:.4f}")
        
        # Learning rate scheduling
        scheduler.step()
        
        # Early stopping
        if val_corr > best_val_corr:
            best_val_corr = val_corr
            patience_counter = 0
            torch.save(model.state_dict(), 'best_mat_model.pt')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return model
