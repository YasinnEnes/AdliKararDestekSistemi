/**
 * FHDSS - Canlı Biyometrik İmza Donanım Soyutlama Katmanı (Hardware Abstraction Layer - HAL)
 * Stylus / Kalem, Dokunmatik Ekran ve Fare için Pointer Events API Adaptörü.
 */

export interface RawBiometricPoint {
  x: number;
  y: number;
  t: number;                  // Milisaniye veya saniye
  pressure: number;           // 0.0 - 1.0 arası normalize
  azimuth_deg?: number;       // 0 - 360 derece kalem yönü
  altitude_deg?: number;      // 0 - 90 derece dikey eğim
  is_pen_down: boolean;
  pointerType?: string;       // "pen", "touch", "mouse"
}

export interface BiometricDynamicFeatures {
  durationSec: number;
  strokeCount: number;
  totalPoints: number;
  meanVelocity: number;       // px/sn
  maxVelocity: number;        // px/sn
  velocityVariance: number;
  meanPressure: number;
  pressureVariance: number;
  pressurePeaksCount: number;
  penDownDurationSec: number;
  inAirRatio: number;
  strokeLengthPx: number;
  hasHardwarePressure: boolean;
}

export interface ISignatureInputProvider {
  startSession(): void;
  endSession(): void;
  recordPoint(point: RawBiometricPoint): void;
  getPoints(): RawBiometricPoint[];
  clear(): void;
  computeFeatures(): BiometricDynamicFeatures;
}

/**
 * Tarayıcı Pointer Events API Sağlayıcısı (Wacom, Apple Pencil, Surface Pen vb.)
 */
export class BrowserPointerProvider implements ISignatureInputProvider {
  private points: RawBiometricPoint[] = [];
  private isCapturing: boolean = false;
  private startTime: number = 0;

  startSession(): void {
    this.isCapturing = true;
    this.points = [];
    this.startTime = performance.now();
  }

  endSession(): void {
    this.isCapturing = false;
  }

  recordPoint(point: RawBiometricPoint): void {
    if (!this.isCapturing && this.points.length === 0) {
      this.isCapturing = true;
      this.startTime = point.t || performance.now();
    }
    this.points.push(point);
  }

  getPoints(): RawBiometricPoint[] {
    return [...this.points];
  }

  clear(): void {
    this.points = [];
    this.isCapturing = false;
    this.startTime = 0;
  }

  computeFeatures(): BiometricDynamicFeatures {
    if (this.points.length < 2) {
      return {
        durationSec: 0,
        strokeCount: 0,
        totalPoints: this.points.length,
        meanVelocity: 0,
        maxVelocity: 0,
        velocityVariance: 0,
        meanPressure: 0.5,
        pressureVariance: 0,
        pressurePeaksCount: 0,
        penDownDurationSec: 0,
        inAirRatio: 0,
        strokeLengthPx: 0,
        hasHardwarePressure: false,
      };
    }

    const t0 = this.points[0].t;
    const tEnd = this.points[this.points.length - 1].t;
    const durationSec = Math.max((tEnd - t0) / 1000, 0.05);

    let totalLengthPx = 0;
    const velocities: number[] = [];
    const pressures: number[] = [];
    let strokeCount = 1;
    let penDownPoints = 0;
    let hasNonZeroPressure = false;

    for (let i = 0; i < this.points.length; i++) {
      const p = this.points[i];
      if (p.pressure > 0 && p.pressure !== 0.5) {
        hasNonZeroPressure = true;
      }
      pressures.push(p.pressure);
      if (p.is_pen_down) penDownPoints++;

      if (i > 0) {
        const prev = this.points[i - 1];
        if (!prev.is_pen_down && p.is_pen_down) {
          strokeCount++;
        }
        const dx = p.x - prev.x;
        const dy = p.y - prev.y;
        const dt = Math.max((p.t - prev.t) / 1000, 0.001);
        const dist = Math.sqrt(dx * dx + dy * dy);
        totalLengthPx += dist;
        velocities.push(dist / dt);
      }
    }

    const meanVelocity = velocities.reduce((a, b) => a + b, 0) / Math.max(velocities.length, 1);
    const maxVelocity = velocities.length ? Math.max(...velocities) : 0;
    const velocityVariance =
      velocities.reduce((acc, v) => acc + Math.pow(v - meanVelocity, 2), 0) / Math.max(velocities.length, 1);

    const meanPressure = pressures.reduce((a, b) => a + b, 0) / Math.max(pressures.length, 1);
    const pressureVariance =
      pressures.reduce((acc, p) => acc + Math.pow(p - meanPressure, 2), 0) / Math.max(pressures.length, 1);

    // Basınç tepe noktaları (peaks)
    let pressurePeaksCount = 0;
    for (let i = 1; i < pressures.length - 1; i++) {
      if (pressures[i] > pressures[i - 1] && pressures[i] > pressures[i + 1] && pressures[i] > meanPressure) {
        pressurePeaksCount++;
      }
    }

    const penDownDurationSec = (penDownPoints / Math.max(this.points.length, 1)) * durationSec;
    const inAirRatio = Math.max(1 - penDownDurationSec / durationSec, 0);

    return {
      durationSec,
      strokeCount,
      totalPoints: this.points.length,
      meanVelocity,
      maxVelocity,
      velocityVariance,
      meanPressure,
      pressureVariance,
      pressurePeaksCount,
      penDownDurationSec,
      inAirRatio,
      strokeLengthPx: totalLengthPx,
      hasHardwarePressure: hasNonZeroPressure,
    };
  }
}

/**
 * Geliştirme ve Donanımsız Testler İçin Mock Dinamik Sağlayıcı
 */
export class MockSignatureProvider implements ISignatureInputProvider {
  private points: RawBiometricPoint[] = [];

  startSession(): void {
    this.points = [];
  }

  endSession(): void {}

  recordPoint(point: RawBiometricPoint): void {
    this.points.push(point);
  }

  getPoints(): RawBiometricPoint[] {
    if (this.points.length > 0) return this.points;

    // Sentetik adli imza yörüngesi üret
    const synthetic: RawBiometricPoint[] = [];
    const now = performance.now();
    for (let i = 0; i < 150; i++) {
      const t = now + i * 15;
      const angle = (i / 150) * Math.PI * 4;
      synthetic.push({
        x: 100 + i * 2.5 + Math.sin(angle) * 35,
        y: 120 + Math.cos(angle) * 25 + (i % 3 === 0 ? 10 : 0),
        t,
        pressure: 0.35 + 0.35 * Math.sin(angle * 0.5),
        is_pen_down: true,
        pointerType: "mock_pen",
      });
    }
    return synthetic;
  }

  clear(): void {
    this.points = [];
  }

  computeFeatures(): BiometricDynamicFeatures {
    const pts = this.getPoints();
    const provider = new BrowserPointerProvider();
    provider.startSession();
    pts.forEach((p) => provider.recordPoint(p));
    return provider.computeFeatures();
  }
}
