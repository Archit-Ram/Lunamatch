/**
 * LunaMatch Custom Exceptions & Error Classes
 */

export class LunaMatchError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'LunaMatchError';
  }
}

export class InvalidImageDataError extends LunaMatchError {
  constructor(reason: string) {
    super(`Invalid ImageData: ${reason}`);
    this.name = 'InvalidImageDataError';
  }
}

export class InvalidCoordinateError extends LunaMatchError {
  constructor(coordinate: { x: number; y: number }, message = 'Coordinate contains NaN, Inf, or out-of-bounds values') {
    super(`InvalidCoordinateError: (${coordinate.x}, ${coordinate.y}) - ${message}`);
    this.name = 'InvalidCoordinateError';
  }
}

export class InsufficientMatchesError extends LunaMatchError {
  constructor(count: number, required: number) {
    super(`Insufficient matches found: ${count} (minimum required is ${required})`);
    this.name = 'InsufficientMatchesError';
  }
}

export class DegenerateGeometryError extends LunaMatchError {
  constructor(reason: string) {
    super(`Degenerate Geometric Configuration: ${reason}`);
    this.name = 'DegenerateGeometryError';
  }
}

export class ModelNotLoadedError extends LunaMatchError {
  constructor(modelName: string) {
    super(`Model weights for ${modelName} are not loaded or unavailable.`);
    this.name = 'ModelNotLoadedError';
  }
}

/**
 * Coordinate validator utility
 */
export function validatePoint(pt: { x: number; y: number }, width?: number, height?: number): boolean {
  if (typeof pt.x !== 'number' || typeof pt.y !== 'number') {
    throw new InvalidCoordinateError(pt, 'Coordinates must be numeric');
  }
  if (Number.isNaN(pt.x) || Number.isNaN(pt.y) || !Number.isFinite(pt.x) || !Number.isFinite(pt.y)) {
    throw new InvalidCoordinateError(pt, 'Coordinates cannot be NaN or Infinite');
  }
  if (width !== undefined && height !== undefined) {
    if (pt.x < -100 || pt.x > width + 100 || pt.y < -100 || pt.y > height + 100) {
      throw new InvalidCoordinateError(pt, `Coordinate out of bounds [0..${width}, 0..${height}]`);
    }
  }
  return true;
}
