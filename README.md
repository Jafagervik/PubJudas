# Anomalydetection for PubDAS with VAE training

## Status

### Data

- [x] Get data from globus
- [ ] Differentiate train and test
- [ ] Split into better sizes

### Setup DDP

- [ ] Without torchrun, make setup and destroy function and make sure we can use multiple gpus on idun

### Config

- [x] Hyperparams stored in json file, try to load

### Trainer

- [ ] Able to initialize trainer
- [ ] save and load from checkpoint file

### Main loop

- [ ] setup full end to end training

### Model

#### VAE

- [x] Encoder
- [x] Decoder
- [x] Elbo loss that works
- [x] forward pass, sampling and recon works

### Anomalydetection

- [ ] Pictures before
- [ ] Pictures after
- [ ] Imgs with found anomlies
- [ ] TP, FP and so on
