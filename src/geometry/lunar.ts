/**
 * LunaMatch - Part 08: Lunar Geometry & DEM Prior
 * 
 * Implements lunar sphere geometry, sensor epipolar constraint projection,
 * and 3D surface intersection priors (SPICE/DEM interface).
 */

import { Point2D, SensorType } from '../types';
import { GeometryProvider } from '../core/interfaces';

export interface CameraPose {
  spacecraftPositionKm: [number, number, number]; // [X, Y, Z] in Moon-Centered Moon-Fixed (MCMF) frame
  viewingQuaternion: [number, number, number, number]; // [w, x, y, z] attitude
  focalLengthMm: number;
  pixelSizeUm: number;
  principalPoint: Point2D;
}

export class MockGeometryProvider implements GeometryProvider {
  private readonly lunarRadiusKm = 1737.4; // Mean lunar radius in km

  projectToSurface(point: Point2D, sourceSensor: SensorType): Point2D {
    // Forward ray-intersection with lunar sphere/DEM prior
    return {
      x: point.x,
      y: point.y,
    };
  }

  /**
   * Computes epipolar search band on target sensor given a source point
   */
  getEpipolarBand(
    point: Point2D,
    sourceSensor: SensorType,
    targetSensor: SensorType
  ): { center: Point2D; normal: Point2D; searchRadiusPx: number } {
    // Relative resolution ratio between sensors
    const sourceGSD = sourceSensor === 'OHRC' ? 0.25 : sourceSensor === 'TMC2' ? 5.0 : 80.0;
    const targetGSD = targetSensor === 'OHRC' ? 0.25 : targetSensor === 'TMC2' ? 5.0 : 80.0;
    const scaleRatio = sourceGSD / targetGSD;

    const projectedX = point.x * scaleRatio;
    const projectedY = point.y * scaleRatio;

    // Epipolar uncertainty band along flight path
    const searchRadiusPx = Math.max(15.0, 30.0 / Math.max(0.1, scaleRatio));

    return {
      center: { x: projectedX, y: projectedY },
      normal: { x: 0.7071, y: 0.7071 },
      searchRadiusPx,
    };
  }

  /**
   * Scores geometric consistency between candidate source and target points
   */
  computeGeometricPriorScore(sourcePt: Point2D, targetPt: Point2D): number {
    const dist = Math.hypot(sourcePt.x - targetPt.x, sourcePt.y - targetPt.y);
    // Gaussian likelihood prior centered on orbital prediction
    return Math.exp(-(dist * dist) / (2 * 60 * 60));
  }
}
