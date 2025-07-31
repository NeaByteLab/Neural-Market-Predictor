import * as ccxt from 'ccxt'

import type { MarketData } from '@/types/index'

/**
 * Market Data Provider using CCXT
 *
 * Fetches real market data from Binance exchange
 * Supports multiple timeframes (1H, 4H, 1D)
 * Handles rate limiting and error management
 *
 * @example
 * ```typescript
 * const provider = new MarketDataProvider()
 * const data = await provider.fetch4H('BTC/USDT', 100)
 * const price = await provider.getLatestPrice('BTC/USDT')
 * ```
 */
export class MarketDataProvider {
  private exchange: ccxt.binance

  constructor() {
    this.exchange = new ccxt.binance({
      enableRateLimit: true,
      options: {
        defaultType: 'future' // Use futures for more data
      }
    })
  }

  /**
   * Fetch OHLCV data from Binance exchange
   *
   * @param symbol - Trading pair symbol (default: 'BTC/USDT')
   * @param timeframe - Time interval (default: '1h')
   * @param limit - Number of data points to fetch (default: 1000)
   * @returns Array of market data with OHLCV values
   */
  async fetchOHLCV(
    symbol: string = 'BTC/USDT',
    timeframe: string = '1h',
    limit: number = 1000
  ): Promise<MarketData[]> {
    try {
      await this.exchange.loadMarkets()
      const ohlcv = await this.exchange.fetchOHLCV(
        symbol,
        timeframe,
        undefined,
        limit
      )
      return ohlcv.map(([timestamp, open, high, low, close, volume]) => ({
        timestamp: timestamp || 0,
        open: open || 0,
        high: high || 0,
        low: low || 0,
        close: close || 0,
        volume: volume || 0
      }))
    } catch (error) {
      console.error('Error fetching market data:', error)
      throw error
    }
  }

  /**
   * Fetch 4-hour timeframe data
   *
   * @param symbol - Trading pair symbol (default: 'BTC/USDT')
   * @param limit - Number of data points to fetch (default: 500)
   * @returns Array of 4H market data
   */
  async fetch4H(
    symbol: string = 'BTC/USDT',
    limit: number = 500
  ): Promise<MarketData[]> {
    return this.fetchOHLCV(symbol, '4h', limit)
  }

  /**
   * Fetch 1-day timeframe data
   *
   * @param symbol - Trading pair symbol (default: 'BTC/USDT')
   * @param limit - Number of data points to fetch (default: 200)
   * @returns Array of 1D market data
   */
  async fetch1D(
    symbol: string = 'BTC/USDT',
    limit: number = 200
  ): Promise<MarketData[]> {
    return this.fetchOHLCV(symbol, '1d', limit)
  }

  /**
   * Get latest price for trading pair
   *
   * @param symbol - Trading pair symbol (default: 'BTC/USDT')
   * @returns Latest price value
   */
  async getLatestPrice(symbol: string = 'BTC/USDT'): Promise<number> {
    try {
      const ticker = await this.exchange.fetchTicker(symbol)
      return ticker.last || 0
    } catch (error) {
      console.error('Error fetching latest price:', error)
      throw error
    }
  }

  /**
   * Get market information for trading pair
   *
   * @param symbol - Trading pair symbol (default: 'BTC/USDT')
   * @returns Market information object
   */
  async getMarketInfo(symbol: string = 'BTC/USDT'): Promise<{
    symbol: string
    base: string
    quote: string
    precision: Record<string, unknown>
    limits: Record<string, unknown>
  }> {
    try {
      await this.exchange.loadMarkets()
      const market = this.exchange.market(symbol)
      return {
        symbol: market.symbol,
        base: market.base,
        quote: market.quote,
        precision: market.precision,
        limits: market.limits
      }
    } catch (error) {
      console.error('Error fetching market info:', error)
      throw error
    }
  }
}
