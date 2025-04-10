# Crop Yield Prediction System

This project predicts crop yields using a **Deep Fusion Neural Network** that integrates **climate data (NASA POWER)**, **soil properties (SoilGrids)**, and historical agricultural data from the **ICRISAT dataset**. The system provides district-level predictions for six major crops: **Rice, Wheat, Maize, Pearl Millet, Finger Millet, and Barley**.

---

## Features

- **Multi-Source Data Integration**:
  - Climate data from [NASA POWER](https://power.larc.nasa.gov/)
  - Soil properties from Soil Health Card (Government of India)
  - Historical crop yield data from ICRISAT (2008–2017)
- **Deep Fusion Neural Network**:
  - Combines temporal (climate), spatial (soil), and crop-specific embeddings
  - Uses attention mechanisms to focus on critical features
- **Spatial Indexing**:
  - Finds the nearest district for user-provided coordinates
- **CLI Interface**:
  - Predict crop yields based on latitude, longitude, and crop type
- **Validation Suite**:
  - Ensures data quality and model performance
