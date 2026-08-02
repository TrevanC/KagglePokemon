# **Technical Specification: Pokémon TCG AI Battle Model Architecture & Feature Pipeline**

**Document Status:** Ready for Engineering Implementation  
**Target Environment:** Kaggle Pokémon TCG AI Battle Challenge (2 vCPUs, 12.2 GiB RAM, CPU-only evaluation)  
**Card Pool Size (**$N\_{\\text{cards}}$**):** 1,267 unique card IDs  
**Inference Latency Budget:** $\< 10 \\text{ ms}$ per turn

## **1\. System Architecture Overview**

This specification details a **Dynamic Action-Scoring Policy Network** designed to rank $k$ legal candidate turn options provided by the game engine. The architecture features a **Modular Board State Encoder** interface, allowing a baseline **MLP State Encoder** to be swapped seamlessly with an attention-based **Transformer State Encoder** without altering downstream action scoring or submission code.  
 \[ Observation Parsing Pipeline \]  
                │  
                ├──► State Feature Dict ──► \[ Board State Encoder (MLP / Transformer) \]  
                │                                           │  
                │                                    State Embedding s (1, 256\)  
                │                                           │  
                │                                           ▼  
                │                                  \[ Virtual Expansion \] ──┐  
                │                                   .expand(k, \-1)         │  
                │                                           │              │  
                └──► Action Matrix A (k, 16\) ──► \[ Action Encoder MLP \]    │  
                                                            │              │  
                                                  Action Embeddings (k, 64)│  
                                                            │              │  
                                                            ▼              ▼  
                                                 \[ Joint Scorer MLP (320 → 1\) \]  
                                                            │  
                                                            ▼  
                                                 \[ Softmax across k choices \]

## **2\. Feature Extraction & Parsing Specification**

### **2.1 Offline/Precomputed Lookup (StaticCardFeatureTable)**

To minimize per-turn CPU latency, card database properties are precomputed into a static lookup matrix $\\mathbf{M}\_{\\text{static}} \\in \\mathbb{R}^{1267 \\times 32}$ during agent initialization.

> * **Source File:** EN\_Card\_Data.csv  
> * **Output Tensor:** self.static\_table of shape (1267, 32\)

| Feature Index Range | Feature Description | Encoding / Normalization |
| :---- | :---- | :---- |
| 0 .. 2 | Category One-Hot | \[Is\_Pokemon, Is\_Trainer, Is\_Energy\] |
| 3 .. 5 | Stage One-Hot | \[Is\_Basic, Is\_Stage1, Is\_Stage2\] |
| 6 | Base HP | $\\frac{\\text{Base HP}}{340.0} \\in \[0.0, 1.0\]$ |
| 7 .. 16 | Element Type One-Hot | 10 Pokémon Types (Grass, Fire, Water, etc.) |
| 17 | Primary Attack Base Damage | $\\frac{\\text{Damage}}{300.0} \\in \[0.0, 1.0\]$ |
| 18 | Retreat Cost | $\\frac{\\text{Retreat Cost}}{4.0} \\in \[0.0, 1.0\]$ |
| 19 .. 31 | Reserved / Secondary Flags | Energy costs, trainer effect flags (padded with zeros) |

### **2.2 Instance-Level Card Vector Construction**

For any specific card instance on the field or in hand, static properties are combined with learned embeddings and live game state:  
> $$\\mathbf{e}\_{\\text{card}} \= \[\\underbrace{\\mathbf{f}\_{\\text{static}}}\_{\\mathbb{R}^{32}} \\,;\\, \\underbrace{\\mathbf{e}\_{\\text{id}}}\_{\\mathbb{R}^{8}} \\,;\\, \\underbrace{\\mathbf{f}\_{\\text{dynamic}}}\_{\\mathbb{R}^{8}}\] \\in \\mathbb{R}^{48} \\xrightarrow{\\text{Linear}(48 \\to 64\) \+ \\text{ReLU}} \\mathbb{R}^{64}$$

> * **Static Vector (**$\\mathbf{f}\_{\\text{static}} \\in \\mathbb{R}^{32}$**):** Looked up via StaticCardFeatureTable\[card\_id\].  
> * **Learned ID Vector (**$\\mathbf{e}\_{\\text{id}} \\in \\mathbb{R}^{8}$**):** Retrieved from nn.Embedding(1267, 8).  
> * **Dynamic Vector (**$\\mathbf{f}\_{\\text{dynamic}} \\in \\mathbb{R}^{8}$**):**  
  1. Current HP Ratio: $\\frac{\\text{current\\\_hp}}{\\text{max\\\_hp}} \\in \[0.0, 1.0\]$ (0.0 if not on field).  
  2. Attached Energy Ratio: $\\min\\left(1.0, \\frac{\\text{attached\\\_energy}}{10.0}\\right)$.  
  3. Status Condition Flags (6-dim multi-hot): \[Poisoned, Burned, Asleep, Paralyzed, Confused, Frozen\].

### **2.3 Online Board State Extraction (obs.current)**

The live board observation is mapped into 4 distinct feature groups:

> 1. **Fixed Field Slots (12 Slots):**  
   * Positions: \[Self Active, Self Bench 1..5, Opp Active, Opp Bench 1..5\].  
   * Unoccupied bench slots are filled with a zero vector of shape (1, 64).  
   * Concatenated shape: (1, 12 \* 64\) \= (1, 768).  
> 2. **Self Hand Cards (Variable Set):**  
   * Mean-pooled embedding across all $N\_{\\text{hand}}$ card vectors in hand.  
   * If hand is empty ($N\_{\\text{hand}} \= 0$), returns a zero vector of shape (1, 64).  
   * Output shape: (1, 64).  
> 3. **Discard Pile Summaries (Self & Opponent):**  
   * Count frequency histogram vector $\\mathbf{v}\_{\\text{hist}} \\in \\mathbb{R}^{1267}$ normalized by $60.0$.  
   * Projected through Linear(1267 \-\> 64\) $\\to$ (1, 64).  
> 4. **Global Game Scalars (16 Dimensions):**  
   * \[self\_prizes/6.0, opp\_prizes/6.0, self\_deck/60.0, opp\_deck/60.0, opp\_hand\_count/20.0, energy\_attached\_this\_turn\_flag, supporter\_played\_this\_turn\_flag, turn\_number/100.0, ... (padded to 16)\].

$$\\text{Raw Board Vector Dimension} \= 768 \+ 64 \+ 64 \+ 16 \= \\mathbf{912} \\text{ dimensions}$$

### **2.4 Action Candidate Extraction (obs.select.option)**

For the turn's $k$ legal candidate actions ($k \\in \[1, 50\]$):

> * **Action Feature Vector (**$\\mathbf{a}\_{\\text{raw}, i} \\in \\mathbb{R}^{16}$**):**  
  1. Option Enum Type One-Hot ($10$ dims: PLAY, ATTACH, EVOLVE, ABILITY, DISCARD, RETREAT, ATTACK, END, etc.).  
  2. Target Normalized Card ID ($\\frac{\\text{card\\\_id}}{1267.0}$).  
  3. Target Slot Index ($\\frac{\\text{slot\\\_idx}}{12.0}$).  
  4. Padded parameter channels ($4$ dims).  
> * Output Action Matrix shape: (k, 16).

### **2.5 Training vs. Inference Feature Extraction Pipeline**

 \[ INFERENCE PIPELINE \]                            \[ TRAINING PIPELINE (Replays/Logs) \]  
  obs.current  ──► Board State Extractor ──► (1, 912\)   Game Log Entry ──► State Reconstruction ──► (1, 912\)  
  obs.select   ──► Action Option Matrix ──► (k, 16\)    Legal Moves at t ──► Candidate Matrix    ──► (k, 16\)  
                                                        Target Move t   ──► Ground Truth Index y ──► integer ∈ \[0, k-1\]

> * **Inference Mode:** Reads obs.current and obs.select.option directly from the live cabt engine.  
> * **Training Mode (Behavioral Cloning / Imitation Learning):**  
  1. Reconstructs state $s\_t$ at time $t$ from game logs.  
  2. Reconstructs all $k$ valid candidate options $\\{a\_1, a\_2, \\dots, a\_k\\}$ available to the player at turn $t$.  
  3. Identifies the index $y \\in \[0, k-1\]$ of the move actually chosen by the expert.  
  4. Optimizes the Cross-Entropy loss over the $k$ candidates:

$$\\mathcal{L}\_{\\text{BC}} \= \-\\log P(a\_y \\mid s\_t) \= \-\\log \\frac{e^{z\_y}}{\\sum\_{j=0}^{k-1} e^{z\_j}}$$

## **3\. Board State Encoder Implementations**

Both encoders implement the abstract interface:  
forward(obs\_dict: dict) \-\> Tensor\[shape=(1, 256)\]

### **3.1 Option A: Baseline MLP Board State Encoder**

Fast, deterministic projection using dense layers.  
Raw State (1, 912\) ──► Linear(912 → 512\) ──► ReLU ──► Linear(512 → 256\) ──► State Embedding s (1, 256\)

#### **Layer Configuration & Shapes:**

> 1. input\_layer: Concatenated raw state tensor (1, 912).  
> 2. projector\_l1: nn.Linear(in\_features=912, out\_features=512)  
> 3. activation\_l1: nn.ReLU()  
> 4. projector\_l2: nn.Linear(in\_features=512, out\_features=256)  
> 5. output: State Embedding $\\mathbf{s} \\in \\mathbb{R}^{1 \\times 256}$.

### **3.2 Option B: Drop-In Transformer Board State Encoder**

Attention-based encoder modeling dynamic relational interactions between cards.  
                        \[ Sequence Assembly \]  
 \[CLS\] Token (1,128) \+ Zone 0  ──► Token 0  ┐  
 12 Field Slots     \+ Zone 1..12 ──► Tokens 1..12 ├──► Sequence Tensor (1, L, 128\)  
 N Hand Cards       \+ Zone 13    ──► Tokens 13..L-1  
 Game Scalars       \+ Zone 14    ──► Token L  ┘  
                                     │  
                                     ▼  
                        \[ Transformer Stack (2 Layers) \]  
                        Multi-Head Self-Attention (d\_model=128, heads=4)  
                                     │  
                                     ▼  
                        \[ Extract Updated \[CLS\] Token \]  
                             cls\_output: (1, 128\)  
                                     │  
                                     ▼  
                        \[ Projection Head \]  
                        Linear(128 → 256\) ──► ReLU ──► Linear(256 → 256\)  
                                     │  
                                     ▼  
                         State Embedding s (1, 256\)

#### **Layer Configuration & Hyperparameters:**

> * **Token Dimension (**$d\_{\\text{model}}$**):** $128$  
> * **Location Embeddings:** nn.Embedding(num\_embeddings=16, embedding\_dim=128)  
  * Zone IDs: 0=\[CLS\], 1=Self Active, 2..6=Self Bench, 7=Opp Active, 8..12=Opp Bench, 13=Hand, 14=Discard, 15=Scalars.  
> * **Classification Token:** nn.Parameter(torch.randn(1, 1, 128))  
> * **Transformer Stack:** nn.TransformerEncoder(encoder\_layer, num\_layers=2)  
  * d\_model: $128$  
  * nhead: $4$ ($d\_{\\text{head}} \= 32$)  
  * dim\_feedforward: $256$ ($2 \\times d\_{\\text{model}}$)  
  * activation: "relu"  
  * batch\_first: True  
> * **Output Projection Head:**  
  * Linear(128, 256\) $\\to$ ReLU() $\\to$ Linear(256, 256\)

## **4\. Action Encoder & Joint Action-Scoring Network**

### **4.1 Action Encoder MLP**

Converts raw candidate action vectors into action embeddings.

> * **Input Matrix:** $\\mathbf{A}\_{\\text{raw}} \\in \\mathbb{R}^{k \\times 16}$  
> * **Architecture:**  
  * Linear(16, 64\) $\\to$ ReLU()  
  * Linear(64, 64\)  
> * **Output Matrix:** $\\mathbf{A}\_{\\text{emb}} \\in \\mathbb{R}^{k \\times 64}$

### **4.2 Joint Scorer & Output Normalization**

Combines board context with action descriptors to evaluate each candidate.  
 State Embedding s (1, 256\)                       Action Embeddings A\_emb (k, 64\)  
            │                                                    │  
            ▼                                                    │  
   .expand(k, \-1)                                                │  
            │                                                    │  
            ▼                                                    │  
 Expanded State (k, 256\)                                         │  
            │                                                    │  
            └─────────────────────────┬──────────────────────────┘  
                                      │  
                                      ▼  
                        torch.cat(\[..., ...\], dim=1)  
                                      │  
                                      ▼  
                         Joint Features Matrix (k, 320\)  
                                      │  
                                      ▼  
                        Joint Scorer MLP (320 → 128 → 1\)  
                        Linear(320 → 128\) ──► ReLU ──► Linear(128 → 1\)  
                                      │  
                                      ▼  
                          Raw Logit Matrix (k, 1\)  
                                      │  
                                      ▼  
                             .squeeze(-1)  
                                      │  
                                      ▼  
                            Logit Vector z (k,)  
                                      │  
                                      ▼  
                           F.softmax(z, dim=0)  
                                      │  
                                      ▼  
                       Action Probabilities P(a|s) (k,)

## **5\. Tensor Dimension & Shape Cheat Sheet**

The table below serves as the definitive reference for tensor shapes across the network pipeline:

| Module / Operation | Input Tensor Name | Input Shape | Transformation / Layer | Output Tensor Name | Output Shape |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **Card Feature Lookup** | card\_id | (N,) | StaticCardFeatureTable \+ nn.Embedding(1267, 8\) | card\_features | (N, 40\) |
| **Card Projection** | card\_instance | (N, 48\) | Linear(48, 64\) \+ ReLU | v\_card | (N, 64\) |
| **Discard Projection** | discard\_hist | (1, 1267\) | Linear(1267, 64\) \+ ReLU | v\_discard | (1, 64\) |
| **MLP State Encoder** | raw\_state | (1, 912\) | Linear(912, 512\) \-\> ReLU \-\> Linear(512, 256\) | state\_emb | **(1, 256\)** |
| **Transformer State Enc.** | token\_sequence | (1, L, 128\) | TransformerEncoder(2L) \-\> \[CLS\] \-\> Linear(128, 256\) | state\_emb | **(1, 256\)** |
| **State Expansion** | state\_emb | (1, 256\) | .expand(k, \-1) | state\_expanded | **(k, 256\)** |
| **Action Encoder** | valid\_actions | (k, 16\) | Linear(16, 64\) \-\> ReLU \-\> Linear(64, 64\) | action\_embs | **(k, 64\)** |
| **Feature Fusion** | \[state, action\] | (k, 256), (k, 64\) | torch.cat(\[state\_expanded, action\_embs\], dim=1) | joint\_features | **(k, 320\)** |
| **Joint Layer 1** | joint\_features | (k, 320\) | Linear(320, 128\) \+ ReLU | joint\_h1 | (k, 128\) |
| **Joint Layer 2** | joint\_h1 | (k, 128\) | Linear(128, 1\) | raw\_scores | (k, 1\) |
| **Dimension Squeeze** | raw\_scores | (k, 1\) | .squeeze(-1) | logits | **(k,)** |
| **Action Softmax** | logits | (k,) | F.softmax(logits, dim=0) | probabilities | **(k,)** |

## **6\. Complete PyTorch Implementation Reference**

import torch  
import torch.nn as nn  
import torch.nn.functional as F

class DynamicActionScorerNetwork(nn.Module):  
    """  
    Complete Pokémon TCG AI Action Scoring Pipeline supporting interchangeable  
    MLP and Transformer state encoders.  
    """  
    def \_\_init\_\_(  
        self,   
        state\_encoder: nn.Module,   
        action\_in\_dim: int \= 16,   
        action\_embed\_dim: int \= 64,   
        state\_embed\_dim: int \= 256  
    ):  
        super().\_\_init\_\_()  
        self.state\_encoder \= state\_encoder  
          
        \# Action Encoder MLP (16 \-\> 64 \-\> 64\)  
        self.action\_encoder \= nn.Sequential(  
            nn.Linear(action\_in\_dim, action\_embed\_dim),  
            nn.ReLU(),  
            nn.Linear(action\_embed\_dim, action\_embed\_dim)  
        )  
          
        \# Joint Scorer Head ( (256 \+ 64\) \-\> 128 \-\> 1 )  
        joint\_in\_dim \= state\_embed\_dim \+ action\_embed\_dim  
        self.joint\_scorer \= nn.Sequential(  
            nn.Linear(joint\_in\_dim, 128),  
            nn.ReLU(),  
            nn.Linear(128, 1\)  
        )

    def forward(self, obs\_dict: dict, valid\_actions\_tensor: torch.Tensor):  
        """  
        obs\_dict: Structured dictionary containing raw board observation  
        valid\_actions\_tensor: Tensor of shape (k, 16\) containing candidate action vectors  
        returns: Logits of shape (k,), Probabilities of shape (k,)  
        """  
        k \= valid\_actions\_tensor.shape\[0\] \# Number of legal candidate actions  
          
        \# 1\. Encode Board State \-\> Shape: (1, 256\)  
        state\_emb \= self.state\_encoder(obs\_dict)  
          
        \# 2\. Virtual State Expansion (zero extra memory) \-\> Shape: (k, 256\)  
        state\_expanded \= state\_emb.expand(k, \-1)  
          
        \# 3\. Encode Action Matrix \-\> Shape: (k, 64\)  
        action\_embs \= self.action\_encoder(valid\_actions\_tensor)  
          
        \# 4\. Feature Fusion via Concatenation \-\> Shape: (k, 320\)  
        joint\_features \= torch.cat(\[state\_expanded, action\_embs\], dim=1)  
          
        \# 5\. Score State-Action Pairs \-\> Shape: (k, 1\) \-\> Squeezed to (k,)  
        logits \= self.joint\_scorer(joint\_features).squeeze(-1)  
          
        \# 6\. Normalize across candidate actions \-\> Shape: (k,)  
        probabilities \= F.softmax(logits, dim=0)  
          
        return logits, probabilities  
