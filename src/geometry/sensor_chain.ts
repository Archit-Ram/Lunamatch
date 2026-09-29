/**
 * LunaMatch - Part 22: Hierarchical Sensor Chain Topology
 *
 * Chandrayaan-2 optical sensors span an ~300x ground-sampling-distance (GSD)
 * range (OHRC 0.25m -> TMC-2 5m -> IIRS 80m). Direct feature matching across
 * that full gap is unreliable: neural matchers are trained on roughly
 * comparable scales, and a single homography/TPS fit over a 300x zoom range
 * amplifies small angular errors into huge ground-truth offsets.
 *
 * Instead of matching endpoints directly, LunaMatch bridges through the
 * intermediate sensor(s) on the canonical fine->coarse chain and composes
 * the per-hop transforms, matching only across adjacent, comparable-scale
 * pairs (OHRC<->TMC-2, TMC-2<->IIRS).
 */

import { SensorType } from '../types';

/** Approximate nadir ground-sampling distance per sensor, in meters/pixel. */
export const SENSOR_GSD_METERS: Record<SensorType, number> = {
  OHRC: 0.25,
  TMC2: 5.0,
  SELENE: 10.0, // Kaguya Terrain Camera, used here only as a coarse reference point
  LRO_NAC: 0.5,
  IIRS: 80.0,
};

/**
 * Canonical fine->coarse bridging order for sensors that participate in
 * hierarchical chaining. Sensors outside this list (e.g. LRO_NAC, SELENE,
 * used as absolute-anchor references rather than pipeline hops) are treated
 * as direct endpoints only - see geometry/lunar.ts for the absolute-anchor
 * concern, which is a separate feature from this chain.
 */
export const CANONICAL_CHAIN_ORDER: SensorType[] = ['OHRC', 'TMC2', 'IIRS'];

/** Scale ratio above which a direct match is considered unreliable and chaining should be preferred. */
export const DEFAULT_CHAIN_SCALE_THRESHOLD = 20;

/** Returns the GSD ratio (>=1) between two sensors. */
export function getScaleRatio(a: SensorType, b: SensorType): number {
  const ga = SENSOR_GSD_METERS[a];
  const gb = SENSOR_GSD_METERS[b];
  if (!ga || !gb) return 1;
  return Math.max(ga, gb) / Math.min(ga, gb);
}

/**
 * Decides whether a source/target sensor pair should be bridged through
 * intermediate sensors rather than matched directly.
 */
export function requiresHierarchicalChaining(
  source: SensorType,
  target: SensorType,
  thresholdRatio: number = DEFAULT_CHAIN_SCALE_THRESHOLD
): boolean {
  if (source === target) return false;
  return getScaleRatio(source, target) > thresholdRatio;
}

/**
 * Resolves the bridging path between two sensors along the canonical
 * fine->coarse chain, e.g. OHRC -> IIRS resolves to [OHRC, TMC2, IIRS].
 * If either endpoint is not part of the canonical chain there is nothing
 * to bridge through, so the direct [source, target] pair is returned.
 */
export function getChainPath(source: SensorType, target: SensorType): SensorType[] {
  if (source === target) return [source];

  const si = CANONICAL_CHAIN_ORDER.indexOf(source);
  const ti = CANONICAL_CHAIN_ORDER.indexOf(target);

  if (si === -1 || ti === -1) {
    return [source, target];
  }

  const [lo, hi] = si < ti ? [si, ti] : [ti, si];
  const path = CANONICAL_CHAIN_ORDER.slice(lo, hi + 1);
  return si < ti ? path : [...path].reverse();
}

/** Human-readable description of a chain path, e.g. "OHRC -> TMC2 -> IIRS". */
export function describeChainPath(path: SensorType[]): string {
  return path.join(' \u2192 ');
}
