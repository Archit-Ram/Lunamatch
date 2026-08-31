/**
 * LunaMatch - Part 18: Spatial Uniformity Optimization
 * 
 * Selects a spatially well-distributed subset of tie-points across an 8x8 spatial grid,
 * preventing degenerate clustering on single high-contrast features.
 */

import { Match, MatchSet } from '../types';
import { UniformityOptimizer } from '../core/interfaces';

export class SpatialUniformitySelector implements UniformityOptimizer {
  private gridDim: number;
  private maxPointsPerCell: number;

  constructor(gridDim: number = 8, maxPointsPerCell: number = 12) {
    this.gridDim = gridDim;
    this.maxPointsPerCell = maxPointsPerCell;
  }

  selectUniformMatches(
    matchSet: MatchSet,
    imageWidth: number,
    imageHeight: number,
    maxTotalMatches: number = 100
  ): MatchSet {
    const rawMatches = matchSet.matches;
    if (rawMatches.length <= maxTotalMatches) {
      return matchSet;
    }

    const cellWidth = imageWidth / this.gridDim;
    const cellHeight = imageHeight / this.gridDim;
    const totalCells = this.gridDim * this.gridDim;

    // Bin matches into grid cells
    const gridBins: Match[][] = Array.from({ length: totalCells }, () => []);

    for (const m of rawMatches) {
      const gx = Math.min(this.gridDim - 1, Math.max(0, Math.floor(m.sourcePoint.x / cellWidth)));
      const gy = Math.min(this.gridDim - 1, Math.max(0, Math.floor(m.sourcePoint.y / cellHeight)));
      const binIdx = gy * this.gridDim + gx;
      gridBins[binIdx].push(m);
    }

    // Sort each bin by confidence & inlier status descending
    for (let i = 0; i < totalCells; i++) {
      gridBins[i].sort((a, b) => {
        const inlierA = a.isInlier ? 1 : 0;
        const inlierB = b.isInlier ? 1 : 0;
        if (inlierA !== inlierB) return inlierB - inlierA;
        return b.confidence - a.confidence;
      });
    }

    // Round-robin selection across active cells to maximize geographic dispersion
    const selectedMatches: Match[] = [];
    let currentRank = 0;

    while (selectedMatches.length < maxTotalMatches) {
      let addedInRound = 0;
      for (let i = 0; i < totalCells; i++) {
        if (selectedMatches.length >= maxTotalMatches) break;
        if (currentRank < gridBins[i].length && currentRank < this.maxPointsPerCell) {
          selectedMatches.push(gridBins[i][currentRank]);
          addedInRound++;
        }
      }
      if (addedInRound === 0) break;
      currentRank++;
    }

    return {
      ...matchSet,
      matches: selectedMatches,
      metadata: {
        ...matchSet.metadata,
        spatialUniformityScore: this.calculateUniformityScore(
          { ...matchSet, matches: selectedMatches },
          imageWidth,
          imageHeight
        ),
      },
    };
  }

  calculateUniformityScore(matchSet: MatchSet, width: number, height: number): number {
    const matches = matchSet.matches.filter((m) => m.isInlier !== false);
    if (matches.length === 0) return 0.0;

    const cellWidth = width / this.gridDim;
    const cellHeight = height / this.gridDim;
    const totalCells = this.gridDim * this.gridDim;
    const cellCounts = new Uint16Array(totalCells);

    for (const m of matches) {
      const gx = Math.min(this.gridDim - 1, Math.max(0, Math.floor(m.sourcePoint.x / cellWidth)));
      const gy = Math.min(this.gridDim - 1, Math.max(0, Math.floor(m.sourcePoint.y / cellHeight)));
      cellCounts[gy * this.gridDim + gx]++;
    }

    let occupiedCells = 0;
    for (let i = 0; i < totalCells; i++) {
      if (cellCounts[i] > 0) occupiedCells++;
    }

    const coverageRatio = occupiedCells / totalCells;

    // Shannon entropy of distribution
    let entropy = 0;
    const totalMatches = matches.length;
    for (let i = 0; i < totalCells; i++) {
      if (cellCounts[i] > 0) {
        const p = cellCounts[i] / totalMatches;
        entropy -= p * Math.log2(p);
      }
    }
    const maxEntropy = Math.log2(occupiedCells || 1) || 1;
    const entropyUniformity = maxEntropy > 0 ? entropy / maxEntropy : 0;

    return Math.min(1.0, Math.max(0.0, 0.5 * coverageRatio + 0.5 * entropyUniformity));
  }
}
