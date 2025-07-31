# Neural Market Predictor

A TypeScript implementation of a neural network for cryptocurrency price prediction using real-time market data from Binance. The system uses a 2-2-1 neural network architecture with sigmoid activation functions to predict future price movements based on historical OHLCV data.

### ✨ Core Features

- **📊 Real Market Data**: Live BTC/USDT data from Binance exchange via CCXT
- **🧠 Neural Network**: 2-2-1 architecture with sigmoid activation and backpropagation
- **🔮 Price Prediction**: Predicts next price based on historical close prices
- **📈 Confidence Scoring**: Calculates prediction confidence based on recent accuracy
- **⚡ TypeScript**: Fully typed implementation with strict TypeScript configuration

---

## 🚀 Quick Start

```bash
# Install dependencies
npm install

# Development mode
npm run dev

# Build and run
npm run build
npm start
```

## 📊 Usage Example

```typescript
import { MarketPredictor, MarketDataProvider } from '@/core'

// Configure neural network
const config = {
  epochs: 60,
  learningRate: 0.1,
  useSimpleBackprop: false
}

// Create predictor and data provider
const predictor = new MarketPredictor(config)
const dataProvider = new MarketDataProvider()

// Fetch real market data
const marketData = await dataProvider.fetch4H('BTC/USDT', 100)

// Add data and train
marketData.forEach(data => predictor.addMarketData(data))
const lossCurve = predictor.train()

// Make prediction
const prediction = predictor.predict()
console.log(`Predicted: $${prediction.predictedOutput}`)
console.log(`Confidence: ${prediction.confidence * 100}%`)
```

---

## 🏗️ Architecture

### 🧠 Neural Network (2-2-1)
- **📥 Input Layer**: 2 nodes (previous close prices)
- **🔗 Hidden Layer**: 2 nodes with sigmoid activation
- **📤 Output Layer**: 1 node (predicted price)
- **🎯 Training**: Backpropagation with MSE loss

### 🔧 Core Components

- **📡 MarketDataProvider**: Fetches real-time data from Binance
- **🎯 MarketPredictor**: High-level prediction interface
- **🧠 NeuralNetwork**: Core neural network implementation

## 📈 Data Sources

- **🏦 Exchange**: Binance (futures)
- **💰 Symbol**: BTC/USDT
- **⏰ Timeframes**: 1H, 4H, 1D
- **📊 Data**: OHLCV (Open, High, Low, Close, Volume)

---

## 🛠️ Development

```bash
# Code quality
npm run lint
npm run format

# Type checking
npm run build

# Testing
npm test
```

## 📁 Project Structure

```
src/
├── core/
│   ├── MarketDataProvider.ts    # Real-time data fetching
│   ├── MarketPredictor.ts       # High-level prediction interface
│   └── NeuralNetwork.ts         # Core neural network
├── types/
│   └── index.ts                 # TypeScript type definitions
└── index.ts                     # Main entry point
```

---

## 🎯 Neural Network Algorithm

The neural network implements the following logic:

1. **Data Normalization**: Scale prices to [0,1] range using max price scaling
2. **Feature Engineering**: Use previous two close prices as input features
3. **Training**: Backpropagation with gradient descent and MSE loss function
4. **Prediction**: Sigmoid activation for price forecasting with denormalization
5. **Confidence**: Error-based confidence calculation using recent prediction accuracy
6. **Weight Initialization**: Seeded random weights for reproducible results

## 📊 Performance

- **Training Time**: ~60 epochs per prediction with configurable learning rate
- **Data Points**: 100-500 historical OHLCV prices from Binance
- **Accuracy**: Varies based on market volatility and trend conditions
- **Confidence**: 0-100% based on recent prediction accuracy
- **Memory Usage**: Efficient with minimal memory footprint
- **Network State**: Saveable/loadable neural network weights and configuration

## 🔧 Configuration

```typescript
interface NeuralNetworkConfig {
  epochs: number        // Training iterations (10-1000)
  learningRate: number  // Gradient descent rate (0.00001-1.0)
  useSimpleBackprop: boolean // Backpropagation algorithm
}
```

---

## 📝 License

MIT License - see LICENSE file for details