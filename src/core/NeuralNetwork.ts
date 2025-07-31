import type {
  LossCurve,
  NeuralNetworkConfig,
  NeuralNetworkState
} from '@/types/index'

/**
 * Neural Network implementation based on Pine Script algorithm
 *
 * Architecture: 2 inputs → 2 hidden → 1 output
 * Uses sigmoid activation and MSE loss function
 * Supports both simple and verbose backpropagation
 *
 * @example
 * ```typescript
 * const nn = new NeuralNetwork({ learningRate: 0.1, epochs: 60 })
 * const loss = nn.train([0.5, 0.3], 0.7)
 * const prediction = nn.predict([0.6, 0.4])
 * ```
 */
export class NeuralNetwork {
  private weights1: number[][] = []
  private weights2: number[][] = []
  private maxScale: number = 0
  private config: NeuralNetworkConfig

  constructor(config: NeuralNetworkConfig) {
    this.config = config
    this.initializeWeights()
  }

  /**
   * Initialize weight matrices with random values
   *
   * Creates w1 (2x2) and w2 (1x2) matrices with seeded random values
   * Called automatically during constructor
   */
  private initializeWeights(): void {
    this.weights1 = this.createRandomMatrix(2, 2)
    this.weights2 = this.createRandomMatrix(1, 2)
  }

  /**
   * Create matrix with random values using seed for reproducibility
   *
   * @param rows - Number of rows in matrix
   * @param cols - Number of columns in matrix
   * @returns Matrix filled with seeded random values
   */
  private createRandomMatrix(rows: number, cols: number): number[][] {
    const matrix: number[][] = []
    const seed = 1337
    for (let i = 0; i < rows; i++) {
      matrix[i] = []
      for (let j = 0; j < cols; j++) {
        matrix[i][j] = this.random(seed + i + j * rows)
      }
    }

    return matrix
  }

  /**
   * Simple random number generator
   *
   * @param seed - Seed value for reproducible randomness
   * @returns Random number between 0 and 1
   */
  private random(seed: number): number {
    const x = Math.sin(seed) * 10000
    return x - Math.floor(x)
  }

  /**
   * Sigmoid activation function
   *
   * @param x - Input value
   * @returns Sigmoid activation between 0 and 1
   */
  private sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-x))
  }

  /**
   * Mean Squared Error loss function
   *
   * @param predicted - Predicted output value
   * @param actual - Actual output value
   * @returns MSE loss value
   */
  private mseLoss(predicted: number, actual: number): number {
    return Math.pow(predicted - actual, 2)
  }

  /**
   * Normalize data between 0 and 1
   *
   * @param data - Raw data value to normalize
   * @returns Normalized value between 0 and 1
   * @throws Error if maxScale is 0 (division by zero)
   */
  public normalize(data: number): number {
    if (this.maxScale === 0) {
      throw new Error('Cannot normalize: maxScale is 0')
    }
    return data / this.maxScale
  }

  /**
   * Revert normalized data back to original scale
   *
   * @param data - Normalized data value
   * @returns Original scale value
   */
  public standardize(data: number): number {
    return data * this.maxScale
  }

  /**
   * Update max scale for normalization
   *
   * @param high - New high value to consider for scaling
   */
  public updateMaxScale(high: number): void {
    this.maxScale = Math.max(this.maxScale, high)
  }

  /**
   * Feed forward through the neural network
   *
   * @param input - Input features array [close1, close2]
   * @returns Tuple of [output, hiddenLayerOutputs]
   * @throws Error if input length is not 2
   */
  private feedforward(input: number[]): [number, number[]] {
    if (!(input.length === 2)) {
      throw new Error('Input must have exactly 2 features')
    }
    const hiddenOut: number[] = []
    for (let i = 0; i < 2; i++) {
      let sum = 0
      for (let j = 0; j < 2; j++) {
        sum += this.weights1[i][j] * input[j]
      }
      hiddenOut.push(this.sigmoid(sum))
    }
    let output = 0
    for (let i = 0; i < 2; i++) {
      output += this.weights2[0][i] * hiddenOut[i]
    }
    output = this.sigmoid(output)
    return [output, hiddenOut]
  }

  /**
   * Simple backpropagation algorithm
   *
   * @param input - Input features array
   * @param actualOutput - Expected output value
   * @param predictedOutput - Predicted output value
   * @param hiddenOut - Hidden layer outputs
   */
  private backpropagationSimple(
    input: number[],
    actualOutput: number,
    predictedOutput: number,
    hiddenOut: number[]
  ): void {
    for (let i = 0; i < 2; i++) {
      this.weights2[0][i] -=
        this.config.learningRate *
        2 *
        (predictedOutput - actualOutput) *
        hiddenOut[i]
    }
    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 2; j++) {
        this.weights1[i][j] -=
          this.config.learningRate *
          2 *
          (predictedOutput - actualOutput) *
          this.weights2[0][i] *
          input[j]
      }
    }
  }

  /**
   * Verbose backpropagation with proper derivatives
   *
   * @param input - Input features array
   * @param actualOutput - Expected output value
   * @param predictedOutput - Predicted output value
   * @param hiddenOut - Hidden layer outputs
   */
  private backpropagationVerbose(
    input: number[],
    actualOutput: number,
    predictedOutput: number,
    hiddenOut: number[]
  ): void {
    const dLossDOutput = 2 * (predictedOutput - actualOutput)
    for (let i = 0; i < 2; i++) {
      const hiddenVal = hiddenOut[i]
      const dLossDW2 = dLossDOutput * hiddenVal
      this.weights2[0][i] -= this.config.learningRate * dLossDW2
    }
    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 2; j++) {
        const inputVal = input[j]
        const w2Val = this.weights2[0][i]
        const hiddenVal = hiddenOut[i]
        const sigmoidDerivative = hiddenVal * (1 - hiddenVal)
        const dLossDW1 = dLossDOutput * w2Val * sigmoidDerivative * inputVal
        this.weights1[i][j] -= this.config.learningRate * dLossDW1
      }
    }
  }

  /**
   * Train the neural network on input data
   *
   * @param input - Normalized input features [close1, close2]
   * @param actualOutput - Normalized expected output
   * @returns Array of loss curve data points
   */
  public train(input: number[], actualOutput: number): LossCurve[] {
    const lossCurve: LossCurve[] = []
    for (let epoch = 1; epoch <= this.config.epochs; epoch++) {
      const [predictedOutput, hiddenOut] = this.feedforward(input)
      const loss = this.mseLoss(predictedOutput, actualOutput)
      console.log(`~~~~ Epoch ${epoch} ~~~~`)
      console.log(`Loss: ${loss}`)
      lossCurve.push({ epoch, loss })
      if (this.config.useSimpleBackprop) {
        this.backpropagationSimple(
          input,
          actualOutput,
          predictedOutput,
          hiddenOut
        )
      } else {
        this.backpropagationVerbose(
          input,
          actualOutput,
          predictedOutput,
          hiddenOut
        )
      }
    }
    return lossCurve
  }

  /**
   * Make a prediction using the trained neural network
   *
   * @param input - Normalized input features [close1, close2]
   * @returns Normalized prediction value
   */
  public predict(input: number[]): number {
    const [predictedOutput] = this.feedforward(input)
    return predictedOutput
  }

  /**
   * Get current network state for saving/loading
   *
   * @returns Current neural network state
   */
  public getState(): NeuralNetworkState {
    return {
      weights1: this.weights1,
      weights2: this.weights2,
      maxScale: this.maxScale
    }
  }

  /**
   * Set network state (for loading saved models)
   *
   * @param state - Neural network state to load
   * @throws Error if state is invalid
   */
  public setState(state: NeuralNetworkState): void {
    if (!(state.weights1 && state.weights2 && state.maxScale !== undefined)) {
      throw new Error('Invalid neural network state')
    }
    this.weights1 = state.weights1
    this.weights2 = state.weights2
    this.maxScale = state.maxScale
  }
}
