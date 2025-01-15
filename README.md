# sim2real_pose_prediction

# (multimodal) transformer for pose prediction in strawberry picking (autnomous driving and vertebrae scanner)

dataset: https://universe.roboflow.com/strawberry-detection-aghjb/strawberry-detection2-iilvu 

| Model                 | Pretraining Dataset | Scale   | #Parameters | .pth Download Link                                                                 |
|-----------------------|---------------------|---------|-------------|-------------------------------------------------------------------------------|
| **Meta-Transformer-B16** | LAION-2B          | Base    | 85M         | [ckpt](https://drive.google.com/file/d/19ahcN2QKknkir_bayhTW5rucuAiX0OXq/view?usp=sharing) |
| **Meta-Transformer-L14** | LAION-2B          | Large   | 302M        | [ckpt](https://drive.google.com/file/d/15EtzCBAQSqmelhdLz6k880A19_RpcX9B/view?usp=drive_link) |


## "Modalities per use case": their dimension
#### Strawberry Picking
"vision": 1024,          
"proprioception": 256,             
"tactile": 128,             

#### Vertebrae Scanning
"pose": 64,             
"ultrasonic": 128,             
"detected_position": 64,             

#### Autonomous Vehicle
"gps": 64,              
"imu": 256,             
"mmwave": 512,             
"lidar": 1024,             
"camera": 1024,             


## Architecture
1. (MT*) **Modality-Specific Tokenization:** Each modality, despite having different dimensions, is tokenized independently into a shared embedding size of 768 to ensure compatibility with the transformer layers and effectively fuse information from different sources to model meaningful interactions across modalities during self attention.
  - Fully connected layer (nn.Linear) maps input to embed_dim.
  - Layer normalization (nn.LayerNorm) standardizes embeddings.
  - GELU activation adds non-linearity.

2. **Temporal Encoding:** Adds temporal (positional) information to tokenized embeddings, essential for predicting time-dependent sequences (e.g., next pose).
  - Sine and cosine functions encode positional information for even and odd indices in the embedding.
    
3. **Fusion of Modalities:** Embeddings from all modalities are stacked, combined (via mean), and passed through the shared transformer.
   
4. **Shared Encoder:** Each token attends to others, enabling the model to learn cross-modal dependencies using multi-head attention with 12 heads and 6 layers. Modalities depend on each other (e.g., how lidar relates to GPS). Therefore, the encoder outputs context-aware embeddings that combine multimodal information.

5. (MT*) **Task-Specific Heads:** Each head is fine-tuned to predict the next pose (x, y, z, α, β, γ) for its respective task (strawberry_picking, vertebrae_scanning, autonomous_vehicle).

6. **Training Workflow:**
  - Loss Function: Mean Squared Error (MSE) compares predicted poses to target values.
  - Optimizer: Adam optimizes the model's parameters.
  - Gradient Update: Backpropagation adjusts parameters to minimize loss.


## Features of This Implementation
- (MT*) **Scalability:** accommodates additional modalities by defining new tokenizers in the modalities dictionary.
- (MT*) **Flexibility:** Task-specific heads allow specialization for diverse use cases (e.g., robotics, medical devices, autonomous systems).
- **Cross-Modal Learning:** Shared transformer encoder enables interactions between modalities, such as tactile-vision alignment for strawberry picking.
- **Temporal Dynamics:** Temporal encoding ensures sequential information is preserved, crucial for pose prediction.
- **Efficient Preprocessing:** Uses pretrained models like ResNet for feature extraction, saving training time.
- **Modularity:** Components like tokenizers, encoder, and task heads are decoupled, simplifying modifications.


## How It Works for Each Task
- **Strawberry Picking:** Combines visual (camera), proprioceptive (joint angles, forces), and tactile (touch sensor) inputs to predict the next position of the robotic arm.
 - **Vertebrae Scanning:** Uses pose, ultrasonic, and detected vertebrae position to predict the device's next position for precise scanning.
- **Autonomous Vehicles:** Integrates lidar, GPS, camera, mmWave radar, and IMU data to predict the vehicle's next movement.

Output: Pose prediction (x, y, z, α, β, γ) for the next time step: torch.Size([32, 6])

This architecture combines multimodal data processing with a transformer-based approach, leveraging state-of-the-art techniques for sequence-to-sequence tasks.

(MT*): inspired by Meta Transformer's architecture 
