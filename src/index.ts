import { MarketDataProvider } from '@core/MarketDataProvider'
import { MarketPredictor } from '@core/MarketPredictor'
import type { MarketData, NeuralNetworkConfig } from '@/types/index'

/**
 * Neural Market Predictor - Main Entry Point
 * Based on Pine Script neural network algorithm with real market data
 */

// Configuration
const config: NeuralNetworkConfig = {
  epochs: 60,
  learningRate: 0.1,
  useSimpleBackprop: false
}

// Create market predictor and data provider
const predictor = new MarketPredictor(config)
const dataProvider = new MarketDataProvider()

// Example usage with real market data
async function runExample(): Promise<void> {
  console.log('🚀 Neural Market Predictor')
  console.log('Based on Pine Script algorithm with real market data\n')

  try {
    // Validate configuration
    if (!(config.learningRate > 0 && config.learningRate <= 1)) {
      throw new Error('Invalid learning rate: must be between 0 and 1')
    }
    if (!(config.epochs > 0 && config.epochs <= 10000)) {
      throw new Error('Invalid epochs: must be between 1 and 10000')
    }
    // Get market info
    const marketInfo = await dataProvider.getMarketInfo()
    console.log('📊 Market Info:')
    console.log(`Symbol: ${marketInfo.symbol}`)
    console.log(`Base: ${marketInfo.base}`)
    console.log(`Quote: ${marketInfo.quote}\n`)

    // Fetch real BTC/USDT data (4H timeframe)
    console.log('📈 Fetching real market data...')
    const marketData = await dataProvider.fetch4H('BTC/USDT', 100)

    if (!(marketData && marketData.length > 0)) {
      throw new Error('Failed to fetch market data')
    }

    console.log(`Fetched ${marketData.length} data points`)
    console.log(`Latest price: $${marketData[marketData.length - 1].close}`)
    console.log(
      `Time range: ${new Date(marketData[0].timestamp)} to ${new Date(marketData[marketData.length - 1].timestamp)}\n`
    )

    // Add data to predictor
    marketData.forEach(data => {
      if (!(data && typeof data.close === 'number' && data.close > 0)) {
        console.warn('Skipping invalid data point:', data)
        return
      }
      predictor.addMarketData(data)
    })

    // Get market stats
    const stats = predictor.getMarketStats()
    console.log('📊 Market Statistics:')
    console.log(`Data Points: ${stats.dataPoints}`)
    console.log(`Max Price: $${stats.maxPrice}`)
    console.log(`Min Price: $${stats.minPrice}`)
    console.log(`Avg Price: $${stats.avgPrice.toFixed(2)}\n`)

    // Train the neural network
    console.log('🧠 Training Neural Network...')
    const lossCurve = predictor.train()

    if (lossCurve) {
      console.log(`Training completed in ${lossCurve.length} epochs`)
      console.log(
        `Final Loss: ${lossCurve[lossCurve.length - 1].loss.toFixed(6)}\n`
      )
    } else {
      console.warn('Training failed: insufficient data')
    }

    // Make prediction
    console.log('🔮 Making Prediction...')
    const prediction = predictor.predict()

    if (prediction) {
      console.log(`Predicted Price: $${prediction.predictedOutput.toFixed(2)}`)
      console.log(`Actual Price: $${prediction.actualOutput}`)
      console.log(`Loss: $${prediction.loss.toFixed(2)}`)
      console.log(`Confidence: ${(prediction.confidence * 100).toFixed(1)}%\n`)
    } else {
      console.warn('Prediction failed: insufficient data')
    }

    // Get network state
    const networkState = predictor.getNetworkState()
    if (networkState) {
      console.log('🔧 Network State:')
      console.log(`Max Scale: ${networkState.maxScale}`)
      console.log(`Weights 1: ${JSON.stringify(networkState.weights1)}`)
      console.log(`Weights 2: ${JSON.stringify(networkState.weights2)}`)
    } else {
      console.warn('Failed to get network state')
    }
  } catch (error) {
    console.error('❌ Error:', error)
    if (error instanceof Error) {
      console.error('Error details:', error.message)
    }
    process.exit(1)
  }
}

// Run example if this file is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  runExample()
}

export { MarketPredictor, MarketDataProvider }
export type { NeuralNetworkConfig, MarketData }
