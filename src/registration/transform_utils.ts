/**
 * LunaMatch - Shared Transform Application Utilities
 *
 * Applies a fitted TransformModel (rigid/affine/homography matrix, or TPS)
 * to a single point, regardless of which model the adaptive estimator chose.
 * Used by the image warper's inverse pass and by the hierarchical chain
 * orchestrator to propagate points through composed transforms.
 */

import { Point2D, TransformModel } from '../types';
import { applyHomographyToPoint } from '../generator/synthetic';
import { AdaptiveTransformEstimator } from './adaptive_transform';

export function applyTransformModel(model: TransformModel, point: Point2D): Point2D {
  if (model.modelType === 'tps' && model.tpsControlPoints) {
    return AdaptiveTransformEstimator.applyTPSTransform(model.tpsControlPoints, point);
  }
  const M = model.matrix || [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];
  return applyHomographyToPoint(M, point);
}
