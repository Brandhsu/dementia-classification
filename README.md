# Dementia Classification

Using deepnets to classify brain pet scans for Alzheimer's disease.

```
.
├── config           # pipeline config (data, model, training)
├── exp              # training results
├── iframe_figures   # embedding plots
├── custom_losses.py # loss functions
├── main.py          # training routine
├── submit.py        # cluster submission
├── eda.ipynb        # viewing model embeddings
├── error.ipynb      # viewingmodel errors
└── tool.ipynb       # tensorboard, etc.
```

Models were trained using the tfcaidm package.