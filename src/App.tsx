/**
 * LunaMatch - Multi-Modal Lunar Image Correspondence Engine
 * Built for ISRO Chandrayaan-2 Optical Data (OHRC, TMC-2, IIRS)
 * SIH 2026 Problem Statement SIH26166
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  GroundTruthData,
  ImageData as LunaImageData,
  RegistrationResult,
  SensorType,
} from './types';
import { generateSyntheticLunarDataset } from './generator/synthetic';
import { IlluminationInvariantNormalizer } from './illumination/invariance';
import { LunaMatchPipeline } from './pipeline/lunamatch';
import { Header } from './components/Header';
import { MetricsBar } from './components/MetricsBar';
import { ViewModeSelector, ViewMode } from './components/ViewModeSelector';
import { DualImageViewer } from './components/DualImageViewer';
import { WarpedOverlayViewer } from './components/WarpedOverlayViewer';
import { SpectralInspector } from './components/SpectralInspector';
import { ControlPanel, PRESET_SCENARIOS, PresetScenario } from './components/ControlPanel';
import { BenchmarkModal } from './components/BenchmarkModal';
import { AblationModal } from './components/AblationModal';
import { TestRunnerModal } from './components/TestRunnerModal';
import { DiagnosticsModal } from './components/DiagnosticsModal';


/**
 * Fetches a grayscale PNG from a URL and converts it to the pipeline's ImageData format.
 */
async function fetchPngAsImageData(
  url: string,
  sensor: SensorType,
  viewLabel: string
): Promise<LunaImageData> {
  const response = await fetch(url);
  const blob = await response.blob();
  const bitmap = await createImageBitmap(blob);

  // Draw to canvas to get pixel data
  const canvas = document.createElement('canvas');
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(bitmap, 0, 0);
  const rawPixels = ctx.getImageData(0, 0, bitmap.width, bitmap.height);

  // Convert RGBA to grayscale Float32Array [0, 1]
  const numPixels = bitmap.width * bitmap.height;
  const pixels = new Float32Array(numPixels);
  for (let i = 0; i < numPixels; i++) {
    // PNGs from ISRO are already grayscale, so R=G=B; just use R channel
    pixels[i] = rawPixels.data[i * 4] / 255.0;
  }

  // Create a grayscale data URL for the viewer
  const dataUrl = canvas.toDataURL('image/png');

  // Build sensor metadata (approximate values from PDS4 XML)
  const sensorMeta = {
    sensorId: sensor,
    spatialResolutionMeters: sensor === 'OHRC' ? 0.25 : 5.0,
    incidenceAngleDeg: sensor === 'TMC2' ? 25 : 0,
    emissionAngleDeg: 0,
    phaseAngleDeg: 25,
    sunAzimuthDeg: 45,
    sunElevationDeg: 25,
    acquisitionTimestamp: '2021-11-22T21:23:22.5689Z',
    productId: `ch2_tmc_${viewLabel}`,
  };

  return {
    id: `real_${sensor}_${viewLabel}_${Date.now()}`,
    pixels,
    width: bitmap.width,
    height: bitmap.height,
    channels: 1,
    dtype: 'float32',
    sensorId: sensor,
    metadata: sensorMeta,
    rawImageDataUrl: dataUrl,
  };
}

export default function App() {
  // Scenario and Sensor state
  const [selectedPreset, setSelectedPreset] = useState<string>('tycho');
  const [sourceSensor, setSourceSensor] = useState<SensorType>('OHRC');
  const [referenceSensor, setReferenceSensor] = useState<SensorType>('TMC2');
  const [matcherChoice, setMatcherChoice] = useState<'fusion' | 'loftr' | 'rift' | 'lightglue' | 'mock'>('fusion');
  const [mockMode, setMockMode] = useState<'perfect' | 'low_noise' | 'high_noise' | 'outlier_heavy' | 'clustered' | 'mixed'>('low_noise');

  // Active Images & Pipeline Result
  const [sourceImage, setSourceImage] = useState<LunaImageData | null>(null);
  const [referenceImage, setReferenceImage] = useState<LunaImageData | null>(null);
  const [normSourceImage, setNormSourceImage] = useState<LunaImageData | null>(null);
  const [normRefImage, setNormRefImage] = useState<LunaImageData | null>(null);
  const [groundTruth, setGroundTruth] = useState<GroundTruthData | null>(null);
  const [registrationResult, setRegistrationResult] = useState<RegistrationResult | null>(null);

  // UI View and Modal states
  const [viewMode, setViewMode] = useState<ViewMode>('matches');
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [isBenchmarkOpen, setIsBenchmarkOpen] = useState<boolean>(false);
  const [isAblationOpen, setIsAblationOpen] = useState<boolean>(false);
  const [isTestsOpen, setIsTestsOpen] = useState<boolean>(false);
  const [isDiagnosticsOpen, setIsDiagnosticsOpen] = useState<boolean>(false);

  // Initialize and generate lunar scenes
  const loadScenario = useCallback(
    async (
      srcSensor: SensorType,
      refSensor: SensorType,
      sunDelta = 180,
      scale = 1.25,
      rot = 15,
      seed = 42
    ) => {
      setIsProcessing(true);

      const dataset = generateSyntheticLunarDataset({
        width: 384,
        height: 384,
        sourceSensor: srcSensor,
        referenceSensor: refSensor,
        sunAzimuthDeg: 45,
        referenceSunAzimuthDeg: (45 + sunDelta) % 360,
        scale,
        rotationDeg: rot,
        seed,
      });

      const normalizer = new IlluminationInvariantNormalizer();
      const nSrc = normalizer.extractInvariantRepresentation(dataset.sourceImage);
      const nRef = normalizer.extractInvariantRepresentation(dataset.referenceImage);

      setSourceImage(dataset.sourceImage);
      setReferenceImage(dataset.referenceImage);
      setNormSourceImage(nSrc);
      setNormRefImage(nRef);
      setGroundTruth(dataset.groundTruth);

      // Execute LunaMatch pipeline
      const pipeline = new LunaMatchPipeline({
        matcher: matcherChoice,
        mockMode,
      });

      const result = await pipeline.registerImages(
        dataset.sourceImage,
        dataset.referenceImage,
        dataset.groundTruth,
        { matcherChoice, mockMode }
      );

      setRegistrationResult(result);
      setIsProcessing(false);
    },
    [matcherChoice, mockMode]
  );

  // Initial Load
  useEffect(() => {
    const preset = PRESET_SCENARIOS[0];
    loadScenario(
      preset.sourceSensor,
      preset.referenceSensor,
      preset.sunAzimuthDelta,
      preset.scaleFactor,
      preset.rotationDeg
    );
  }, []);

  /**
   * Load real satellite images from PNG files and run the pipeline.
   */
  const loadRealImages = useCallback(
    async (preset: PresetScenario) => {
      if (!preset.sourceImageUrl || !preset.referenceImageUrl) return;
      setIsProcessing(true);

      try {
        // Fetch both PNGs and decode them via an offscreen canvas
        const [srcImg, refImg] = await Promise.all([
          fetchPngAsImageData(preset.sourceImageUrl, preset.sourceSensor, 'Aft (-25°)'),
          fetchPngAsImageData(preset.referenceImageUrl, preset.referenceSensor, 'Nadir (0°)'),
        ]);

        const normalizer = new IlluminationInvariantNormalizer();
        const nSrc = normalizer.extractInvariantRepresentation(srcImg);
        const nRef = normalizer.extractInvariantRepresentation(refImg);

        setSourceImage(srcImg);
        setReferenceImage(refImg);
        setNormSourceImage(nSrc);
        setNormRefImage(nRef);
        setGroundTruth(null); // No ground truth for real data

        const pipeline = new LunaMatchPipeline({
          matcher: matcherChoice,
          mockMode,
        });

        const result = await pipeline.registerImages(
          srcImg,
          refImg,
          undefined as any,
          { matcherChoice, mockMode }
        );

        setRegistrationResult(result);
      } catch (err) {
        console.error('Failed to load real images:', err);
      }
      setIsProcessing(false);
    },
    [matcherChoice, mockMode]
  );

  const handleSelectPreset = (preset: PresetScenario) => {
    setSelectedPreset(preset.id);
    setSourceSensor(preset.sourceSensor);
    setReferenceSensor(preset.referenceSensor);
    if (preset.isRealData) {
      loadRealImages(preset);
    } else {
      loadScenario(
        preset.sourceSensor,
        preset.referenceSensor,
        preset.sunAzimuthDelta,
        preset.scaleFactor,
        preset.rotationDeg,
        Date.now() % 10000
      );
    }
  };

  const handleRunCorrespondence = () => {
    const currentPreset = PRESET_SCENARIOS.find((p) => p.id === selectedPreset);
    if (currentPreset?.isRealData) {
      loadRealImages(currentPreset);
    } else {
      loadScenario(
        sourceSensor,
        referenceSensor,
        currentPreset?.sunAzimuthDelta || 120,
        currentPreset?.scaleFactor || 1.25,
        currentPreset?.rotationDeg || 15,
        Date.now() % 10000
      );
    }
  };

  const handleResetScene = () => {
    handleRunCorrespondence();
  };

  const hasIIRSSensor = sourceSensor === 'IIRS' || referenceSensor === 'IIRS';

  return (
    <div className="flex flex-col h-screen w-screen bg-slate-950 text-slate-100 font-sans overflow-hidden">
      {/* Top Header */}
      <Header
        activeMatcher={matcherChoice}
        isProcessing={isProcessing}
        onOpenBenchmarks={() => setIsBenchmarkOpen(true)}
        onOpenAblation={() => setIsAblationOpen(true)}
        onOpenTests={() => setIsTestsOpen(true)}
        onOpenDiagnostics={() => setIsDiagnosticsOpen(true)}
      />

      {/* Scientific Metrics Bar */}
      <MetricsBar
        status={registrationResult?.status || 'idle'}
        failureReason={registrationResult?.failureReason}
        metrics={registrationResult?.metrics}
        transform={registrationResult?.transform}
        sourceSensor={sourceSensor}
        referenceSensor={referenceSensor}
      />

      {/* Main Workspace Area */}
      <div className="flex-1 flex flex-col lg:flex-row overflow-hidden">
        {/* Left Mission Control Panel */}
        <ControlPanel
          selectedPreset={selectedPreset}
          onSelectPreset={handleSelectPreset}
          sourceSensor={sourceSensor}
          setSourceSensor={setSourceSensor}
          referenceSensor={referenceSensor}
          setReferenceSensor={setReferenceSensor}
          matcherChoice={matcherChoice}
          setMatcherChoice={setMatcherChoice}
          mockMode={mockMode}
          setMockMode={setMockMode}
          isProcessing={isProcessing}
          onRunCorrespondence={handleRunCorrespondence}
          onResetScene={handleResetScene}
        />

        {/* Central Visualization Viewport */}
        <div className="flex-1 flex flex-col min-w-0 bg-slate-950 overflow-hidden">
          {/* View Mode Selector Tabs */}
          <ViewModeSelector
            currentMode={viewMode}
            onSelectMode={setViewMode}
            hasIIRSSensor={hasIIRSSensor}
          />

          {/* Canvas Render Area */}
          <div className="flex-1 overflow-hidden relative">
            {sourceImage && referenceImage && (
              <>
                {viewMode === 'spectral' ? (
                  <SpectralInspector
                    imageWidth={sourceImage.width}
                    imageHeight={sourceImage.height}
                  />
                ) : viewMode === 'warped' && registrationResult?.registeredImage ? (
                  <WarpedOverlayViewer
                    referenceImage={referenceImage}
                    registeredSourceImage={registrationResult.registeredImage}
                    transform={registrationResult.transform}
                  />
                ) : (
                  <DualImageViewer
                    sourceImage={sourceImage}
                    referenceImage={referenceImage}
                    normSourceImage={normSourceImage || undefined}
                    normRefImage={normRefImage || undefined}
                    matches={
                      viewMode === 'raw'
                        ? registrationResult?.rawMatches
                        : registrationResult?.uniformMatches || registrationResult?.fusedMatches
                    }
                    transform={registrationResult?.transform}
                    viewMode={viewMode}
                  />
                )}
              </>
            )}
          </div>
        </div>
      </div>

      {/* Modal Dialogs */}
      <BenchmarkModal
        isOpen={isBenchmarkOpen}
        onClose={() => setIsBenchmarkOpen(false)}
      />

      <AblationModal
        isOpen={isAblationOpen}
        onClose={() => setIsAblationOpen(false)}
      />

      <TestRunnerModal
        isOpen={isTestsOpen}
        onClose={() => setIsTestsOpen(false)}
      />

      <DiagnosticsModal
        isOpen={isDiagnosticsOpen}
        onClose={() => setIsDiagnosticsOpen(false)}
        diagnostics={registrationResult?.diagnostics}
        transform={registrationResult?.transform}
      />
    </div>
  );
}
