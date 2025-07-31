/**
 * Configuration for neural network training
 */
export interface NeuralNetworkConfig {
  /** Learning rate for gradient descent (0.00001 to 1.0) */
  learningRate: number
  /** Number of training epochs (10 to 1000) */
  epochs: number
  /** Use simple vs verbose backpropagation */
  useSimpleBackprop: boolean
}

/**
 * Training data for neural network
 */
export interface TrainingData {
  /** Input features (normalized values) */
  input: number[]
  /** Expected output (normalized value) */
  actualOutput: number
}

/**
 * Result of neural network prediction
 */
export interface PredictionResult {
  /** Predicted price value */
  predictedOutput: number
  /** Actual price value */
  actualOutput: number
  /** Mean squared error loss */
  loss: number
  /** Prediction confidence (0-1) */
  confidence: number
}

/**
 * Loss curve data point during training
 */
export interface LossCurve {
  /** Training epoch number */
  epoch: number
  /** Loss value at this epoch */
  loss: number
}

/**
 * Neural network internal state for saving/loading
 */
export interface NeuralNetworkState {
  /** First layer weights (2x2 matrix) */
  weights1: number[][]
  /** Second layer weights (1x2 matrix) */
  weights2: number[][]
  /** Maximum scale for normalization */
  maxScale: number
}

/**
 * Market data structure from exchange
 */
export interface MarketData {
  /** Unix timestamp in milliseconds */
  timestamp: number
  /** Opening price */
  open: number
  /** Highest price in period */
  high: number
  /** Lowest price in period */
  low: number
  /** Closing price */
  close: number
  /** Trading volume */
  volume: number
}
