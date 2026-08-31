/**
 * LunaMatch - Part 03: PDS4 Data Ingestion & Sensor Adapters
 * 
 * Ingests Chandrayaan-2 (OHRC, TMC-2, IIRS) and reference (LRO NAC, SELENE) products,
 * parses PDS4 XML labels/metadata, and converts to unified ImageData format.
 */

import { ImageData, SensorMetadata, SensorType } from '../types';

export interface PDS4LabelMetadata {
  productId: string;
  instrumentId: string;
  targetName: string;
  startTimeUtc: string;
  stopTimeUtc: string;
  spatialResolutionMeters: number;
  sunAzimuthDeg: number;
  sunElevationDeg: number;
  incidenceAngleDeg: number;
  emissionAngleDeg: number;
  phaseAngleDeg: number;
  centerLatitude: number;
  centerLongitude: number;
  lines: number;
  samples: number;
  bands?: number;
}

export interface SensorAdapter {
  readonly sensorId: SensorType;
  parseMetadata(rawMetadataTextOrJson: string | Record<string, any>): SensorMetadata;
  toImageData(pixelArray: Float32Array | Uint8Array | Uint16Array, width: number, height: number, meta: SensorMetadata): ImageData;
}

export class OHRCAdapter implements SensorAdapter {
  readonly sensorId: SensorType = 'OHRC';

  parseMetadata(input: string | Record<string, any>): SensorMetadata {
    const raw = typeof input === 'string' ? JSON.parse(input) : input;
    return {
      sensorId: 'OHRC',
      productId: raw.productId || 'CH2_OHRC_GENERIC',
      spatialResolutionMeters: raw.spatialResolutionMeters || 0.25, // OHRC 0.25m
      incidenceAngleDeg: raw.incidenceAngleDeg ?? 45,
      emissionAngleDeg: raw.emissionAngleDeg ?? 0,
      phaseAngleDeg: raw.phaseAngleDeg ?? 45,
      sunAzimuthDeg: raw.sunAzimuthDeg ?? 60,
      sunElevationDeg: raw.sunElevationDeg ?? 35,
      centerLatitude: raw.centerLatitude ?? -18.2,
      centerLongitude: raw.centerLongitude ?? 82.4,
      spacecraftAltitudeKm: raw.spacecraftAltitudeKm ?? 100,
    };
  }

  toImageData(pixels: Float32Array, width: number, height: number, meta: SensorMetadata): ImageData {
    return {
      id: `ohrc_${Date.now()}`,
      pixels,
      width,
      height,
      channels: 1,
      dtype: 'float32',
      sensorId: 'OHRC',
      metadata: meta,
      mask: new Uint8Array(width * height).fill(1),
    };
  }
}

export class TMC2Adapter implements SensorAdapter {
  readonly sensorId: SensorType = 'TMC2';

  parseMetadata(input: string | Record<string, any>): SensorMetadata {
    const raw = typeof input === 'string' ? JSON.parse(input) : input;
    return {
      sensorId: 'TMC2',
      productId: raw.productId || 'CH2_TMC2_GENERIC',
      spatialResolutionMeters: raw.spatialResolutionMeters || 5.0, // TMC-2 5.0m
      incidenceAngleDeg: raw.incidenceAngleDeg ?? 50,
      emissionAngleDeg: raw.emissionAngleDeg ?? 12, // Fore/Nadir/Aft stereo angles
      phaseAngleDeg: raw.phaseAngleDeg ?? 42,
      sunAzimuthDeg: raw.sunAzimuthDeg ?? 70,
      sunElevationDeg: raw.sunElevationDeg ?? 30,
      centerLatitude: raw.centerLatitude ?? -18.2,
      centerLongitude: raw.centerLongitude ?? 82.4,
      spacecraftAltitudeKm: raw.spacecraftAltitudeKm ?? 100,
    };
  }

  toImageData(pixels: Float32Array, width: number, height: number, meta: SensorMetadata): ImageData {
    return {
      id: `tmc2_${Date.now()}`,
      pixels,
      width,
      height,
      channels: 1,
      dtype: 'float32',
      sensorId: 'TMC2',
      metadata: meta,
      mask: new Uint8Array(width * height).fill(1),
    };
  }
}

export class IIRSAdapter implements SensorAdapter {
  readonly sensorId: SensorType = 'IIRS';

  parseMetadata(input: string | Record<string, any>): SensorMetadata {
    const raw = typeof input === 'string' ? JSON.parse(input) : input;
    return {
      sensorId: 'IIRS',
      productId: raw.productId || 'CH2_IIRS_GENERIC',
      spatialResolutionMeters: raw.spatialResolutionMeters || 80.0, // IIRS 80m
      incidenceAngleDeg: raw.incidenceAngleDeg ?? 40,
      emissionAngleDeg: raw.emissionAngleDeg ?? 0,
      phaseAngleDeg: raw.phaseAngleDeg ?? 40,
      sunAzimuthDeg: raw.sunAzimuthDeg ?? 55,
      sunElevationDeg: raw.sunElevationDeg ?? 40,
      bandsCount: raw.bandsCount || 250,
      wavelengthRangeNm: raw.wavelengthRangeNm || [800, 5000],
      centerLatitude: raw.centerLatitude ?? -18.2,
      centerLongitude: raw.centerLongitude ?? 82.4,
      spacecraftAltitudeKm: raw.spacecraftAltitudeKm ?? 100,
    };
  }

  toImageData(pixels: Float32Array, width: number, height: number, meta: SensorMetadata): ImageData {
    return {
      id: `iirs_${Date.now()}`,
      pixels,
      width,
      height,
      channels: 3, // False-color representation for 2D matching
      bands: Array.from({ length: 250 }, (_, i) => 800 + i * ((5000 - 800) / 249)),
      dtype: 'float32',
      sensorId: 'IIRS',
      metadata: meta,
      mask: new Uint8Array(width * height).fill(1),
    };
  }
}

/**
 * Sensor Registry
 */
export class SensorRegistry {
  private static adapters: Map<SensorType, SensorAdapter> = new Map([
    ['OHRC', new OHRCAdapter()],
    ['TMC2', new TMC2Adapter()],
    ['IIRS', new IIRSAdapter()],
  ]);

  static getAdapter(sensor: SensorType): SensorAdapter {
    const adapter = this.adapters.get(sensor);
    if (!adapter) {
      throw new Error(`Sensor adapter not registered for: ${sensor}`);
    }
    return adapter;
  }
}
