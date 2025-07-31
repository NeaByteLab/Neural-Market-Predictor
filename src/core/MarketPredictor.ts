import type {
  LossCurve,
  MarketData,
  NeuralNetworkConfig,
  NeuralNetworkState,
  PredictionResult
} from '@/types/index'

import { NeuralNetwork } from '@core/NeuralNetwork'

/**
 * Market Predictor using Neural Network
 *
 * Wraps the neural network with market data handling
 * Provides high-level interface for price prediction
 * Supports real-time training and prediction
 *
 * @example
 * ```typescript
 * const predictor = new MarketPredictor(config)
 * predictor.addMarketData({ close: 45000, high: 45500, low: 44800, volume: 1000, timestamp: Date.now() })
 * const prediction = predictor.predict()
 * ```
 */
export class MarketPredictor {
  private neuralNetwork: NeuralNetwork
  private marketData: MarketData[] = []

  constructor(config: NeuralNetworkConfig) {
    this.neuralNetwork = new NeuralNetwork(config)
  }

  /**
   * Add market data point to the predictor
   *
   * @param data - Market data with OHLCV values
   */
  public addMarketData(data: MarketData): void {
    this.marketData.push(data)
    this.neuralNetwork.updateMaxScale(data.high)
  }

  /**
   * Get training data from market data
   *
   * @returns Training data object or null if insufficient data
   */
  private getTrainingData(): { input: number[]; actualOutput: number } | null {
    if (this.marketData.length < 3) {
      return null
    }
    const current = this.marketData[this.marketData.length - 1]
    const prev1 = this.marketData[this.marketData.length - 2]
    const prev2 = this.marketData[this.marketData.length - 3]
    const input = [
      this.neuralNetwork.normalize(prev1.close),
      this.neuralNetwork.normalize(prev2.close)
    ]
    const actualOutput = this.neuralNetwork.normalize(current.close)
    return { input, actualOutput }
  }

  /**
   * Train the neural network on current market data
   *
   * @returns Array of loss curve data points or null if insufficient data
   */
  public train(): LossCurve[] | null {
    const trainData = this.getTrainingData()
    if (!trainData) {
      console.log('Not enough data for training')
      return null
    }
    console.log('Training neural network...')
    console.log(`Input: [${trainData.input.map(x => x.toFixed(4)).join(', ')}]`)
    console.log(`Actual Output: ${trainData.actualOutput.toFixed(4)}`)
    return this.neuralNetwork.train(trainData.input, trainData.actualOutput)
  }

  /**
   * Make a prediction for the next price
   *
   * @returns Prediction result with price, loss, and confidence or null if insufficient data
   */
  public predict(): PredictionResult | null {
    if (!(this.marketData.length >= 2)) {
      return null
    }
    const prev1 = this.marketData[this.marketData.length - 1]
    const prev2 = this.marketData[this.marketData.length - 2]
    const input = [
      this.neuralNetwork.normalize(prev1.close),
      this.neuralNetwork.normalize(prev2.close)
    ]
    const predNormalized = this.neuralNetwork.predict(input)
    const predOutput = this.neuralNetwork.standardize(predNormalized)
    const confidence = this.calculateConfidence()
    return {
      predictedOutput: predOutput,
      actualOutput: prev1.close,
      loss: Math.abs(predOutput - prev1.close),
      confidence
    }
  }

  /**
   * Calculate prediction confidence based on recent accuracy
   *
   * @returns Confidence value between 0 and 1
   */
  private calculateConfidence(): number {
    if (!(this.marketData.length >= 5)) {
      return 0.5
    }
    let totalError = 0
    const recentData = this.marketData.slice(-5)
    for (let i = 2; i < recentData.length; i++) {
      const input = [
        this.neuralNetwork.normalize(recentData[i - 1].close),
        this.neuralNetwork.normalize(recentData[i - 2].close)
      ]
      const predicted = this.neuralNetwork.predict(input)
      const predictedPrice = this.neuralNetwork.standardize(predicted)
      const actualPrice = recentData[i].close
      totalError += Math.abs(predictedPrice - actualPrice) / actualPrice
    }
    const avgError = totalError / (recentData.length - 2)
    const confidence = Math.max(0, 1 - avgError)
    return Math.min(1, confidence)
  }

  /**
   * Get market statistics from current data
   *
   * @returns Market statistics object
   */
  public getMarketStats(): {
    dataPoints: number
    maxPrice: number
    minPrice: number
    avgPrice: number
    } {
    if (!(this.marketData.length > 0)) {
      return { dataPoints: 0, maxPrice: 0, minPrice: 0, avgPrice: 0 }
    }
    const prices = this.marketData.map(d => d.close)
    const maxPrice = Math.max(...prices)
    const minPrice = Math.min(...prices)
    const avgPrice =
      prices.reduce((sum, price) => sum + price, 0) / prices.length
    return {
      dataPoints: this.marketData.length,
      maxPrice,
      minPrice,
      avgPrice
    }
  }

  /**
   * Get neural network state for saving/loading
   *
   * @returns Current neural network state
   */
  public getNetworkState(): NeuralNetworkState {
    return this.neuralNetwork.getState()
  }

  /**
   * Set neural network state
   *
   * @param state - Neural network state to load
   * @throws Error if state is invalid
   */
  public setNetworkState(state: NeuralNetworkState): void {
    if (!(state && typeof state === 'object')) {
      throw new Error('Invalid network state')
    }
    this.neuralNetwork.setState(state)
  }

  /**
   * Clear all market data from predictor
   */
  public clearData(): void {
    this.marketData = []
  }
}
