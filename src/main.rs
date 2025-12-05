use macroquad::prelude::*;
use std::collections::HashMap;

const TIME_LENGTH: f32 = 40.0; // Extended for better visibility
const PARTICLE_SPEED: f32 = 0.05; // Slow speed for observation
const MAX_RADIUS: f32 = 10.0; // Not strictly used in new model but kept for sizing
const GRID_SIZE: f32 = 0.5;
const Z_SPREAD: f32 = 1.5; // Transverse thickness to let tilted slits skew the pattern

// Physics Constants
const LAMBDA: f32 = 3.0; // Amplitude of Quantum Time oscillation (reduced to avoid bias)
const OMEGA: f32 = 0.6;  // Frequency of Quantum Time oscillation (slower, keeps symmetry)
const TRANSVERSE_AMPLITUDE: f32 = 8.0; // Amplitude of Y oscillation
const TRANSVERSE_FREQUENCY: f32 = 1.8; // Frequency of Y oscillation

struct Particle {
    pos: Vec3,
    t_c: f32, // Classical Time
    phase_offset: f32,
    trail: Vec<Vec3>,
    slit_passed: i8, // 0 = not yet, 1 = slit 1 (negative Y), 2 = slit 2 (positive Y)
}

struct DiagnosticStats {
    count: u32,
    sum_y: f32,
    positive: u32,
}

fn grid_key(y: f32, z: f32, grid_size: f32) -> (i32, i32) {
    ((y / grid_size).round() as i32, (z / grid_size).round() as i32)
}

#[macroquad::main("Quantum Time Simulation")]
async fn main() {
    let mut cam_dist = 60.0;
    let mut cam_yaw: f32 = std::f32::consts::PI / 2.0; // Look at X-Y plane
    let mut cam_pitch: f32 = 0.0;
    let target = vec3(TIME_LENGTH / 2.0, 0.0, 0.0);

    let mut particles: Vec<Particle> = Vec::new();
    let mut intensity_map: HashMap<(i32, i32), u32> = HashMap::new();
    let mut max_intensity = 1u32;
    let mut spawn_timer = 0.0;
    
    // Analysis State
    let mut analyzing = false;
    let mut paused = false; // True after analysis/sweep completes
    let mut analysis_hits = 0;
    let mut total_fired = 0u32;
    let target_hits = 1_000_000; // 1 Million Particles for single analysis
    let sweep_target_hits = 100_000; // 100k per sweep step
    let mut metrics_text = String::new();
    let mut line_plot_points: Vec<Vec2> = Vec::new();
    let mut diag_stats = DiagnosticStats { count: 0, sum_y: 0.0, positive: 0 };
    
    // Mirror State
    let mut mirrored = false;
    let mut wavefunction_mode = false;
    
    // Slit Tracking
    let mut slit1_hits = 0u32;
    let mut slit2_hits = 0u32;
    let mut slit_enabled = true;
    
    // Sweep State
    let mut sweep_mode = false;
    let mut current_slit_width = 0.2f32;
    let mut sweep_results: Vec<(f32, f32, f32, f32)> = Vec::new(); // (width, fwhm, contrast, symmetry)
    
    // Phase Distribution State: 0=Uniform, 1=Sinusoidal, 2=Gaussian(π)
    let mut phase_dist_mode = 0u8;
    let mut phase_sweep_mode = false;
    let mut phase_sweep_results: Vec<(String, f32, f32, u32, u32)> = Vec::new(); // (name, contrast, symmetry, slit1, slit2)
    
    // Slit Angle State
    let mut slit_angle: f32 = 0.0; // In degrees
    let angle_sweep_values: [f32; 4] = [0.0, 15.0, -15.0, 30.0]; // Angles to test
    let mut angle_sweep_mode = false;
    let mut angle_index = 0usize;
    let mut angle_sweep_results: Vec<(f32, f32, f32, f32, f32, f32, f32, f32)> = Vec::new(); // (angle, centroid_y, centroid_z, pattern_angle_deg, yz_corr, contrast, symmetry, slit_ratio)
    let mut diagnostic_mode = false;

    loop {
        clear_background(BLACK);

        // Camera controls
        if is_mouse_button_down(MouseButton::Left) {
            let delta = mouse_delta_position();
            cam_yaw -= delta.x * 1.0;
            cam_pitch += delta.y * 1.0;
            cam_pitch = cam_pitch.clamp(-1.5, 1.5);
        }
        
        let wheel = mouse_wheel().1;
        if wheel != 0.0 {
            cam_dist -= wheel * 2.0;
            cam_dist = cam_dist.clamp(5.0, 200.0);
        }

        let cam_x = target.x + cam_dist * cam_pitch.cos() * cam_yaw.cos();
        let cam_y = target.y + cam_dist * cam_pitch.sin();
        let cam_z = target.z + cam_dist * cam_pitch.cos() * cam_yaw.sin();

        let camera = Camera3D {
            position: vec3(cam_x, cam_y, cam_z),
            target,
            up: vec3(0.0, 0.0, 1.0), // Z is up now
            ..Default::default()
        };

        set_camera(&camera);

        // Draw axes
        draw_line_3d(vec3(0.0, 0.0, 0.0), vec3(TIME_LENGTH + 5.0, 0.0, 0.0), RED);
        draw_line_3d(vec3(0.0, -20.0, 0.0), vec3(0.0, 20.0, 0.0), GREEN);
        draw_line_3d(vec3(0.0, 0.0, -10.0), vec3(0.0, 0.0, 10.0), BLUE);

        // Draw Double Slit Barrier
        let barrier_x = TIME_LENGTH / 2.0;
        let slit_width = if sweep_mode { current_slit_width } else { 2.0 };
        let slit_separation = 6.0;
        let barrier_height = 20.0;
        let barrier_width = 30.0;
        
    
    let s2 = slit_separation / 2.0;
    let w2 = slit_width / 2.0;
    
    // Barrier Parts (Gray)
    if slit_enabled {
        let y_start_1: f32 = -barrier_width / 2.0;
        let y_end_1: f32 = -s2 - w2;
        let center_y_1 = (y_start_1 + y_end_1) / 2.0;
        let size_y_1 = (y_end_1 - y_start_1).abs();
        draw_cube(vec3(barrier_x, center_y_1, 0.0), vec3(0.5, size_y_1, barrier_height), None, GRAY);

        let y_start_2: f32 = -s2 + w2;
        let y_end_2: f32 = s2 - w2;
        let center_y_2 = (y_start_2 + y_end_2) / 2.0;
        let size_y_2 = (y_end_2 - y_start_2).abs();
        draw_cube(vec3(barrier_x, center_y_2, 0.0), vec3(0.5, size_y_2, barrier_height), None, GRAY);

        let y_start_3: f32 = s2 + w2;
        let y_end_3: f32 = barrier_width / 2.0;
        let center_y_3 = (y_start_3 + y_end_3) / 2.0;
        let size_y_3 = (y_end_3 - y_start_3).abs();
        draw_cube(vec3(barrier_x, center_y_3, 0.0), vec3(0.5, size_y_3, barrier_height), None, GRAY);
    }


        // Spawn Logic
        if analyzing {
            // Fast batch spawn for analysis
            let current_target = if sweep_mode || phase_sweep_mode || angle_sweep_mode { sweep_target_hits } else { target_hits };
            if analysis_hits < current_target {
                for _ in 0..2000 { // Spawn 2000 per frame for speed
                    // Generate phase based on distribution mode
                    let phase = match phase_dist_mode {
                        0 => rand::gen_range(0.0, 2.0 * std::f32::consts::PI), // Uniform
                        1 => { // Sinusoidal: PDF ~ 1 + sin(x), use rejection sampling
                            loop {
                                let x = rand::gen_range(0.0, 2.0 * std::f32::consts::PI);
                                let y = rand::gen_range(0.0, 2.0);
                                if y < 1.0 + x.sin() {
                                    break x;
                                }
                            }
                        },
                        2 => { // Gaussian centered at π, σ = π/3
                            let u1: f32 = rand::gen_range(0.0001, 1.0);
                            let u2: f32 = rand::gen_range(0.0, 1.0);
                            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                            let sigma = std::f32::consts::PI / 3.0;
                            (std::f32::consts::PI + z * sigma).rem_euclid(2.0 * std::f32::consts::PI)
                        },
                        _ => rand::gen_range(0.0, 2.0 * std::f32::consts::PI),
                    };
                    let z0 = rand::gen_range(-Z_SPREAD, Z_SPREAD);
                    particles.push(Particle {
                        pos: vec3(0.0, 0.0, z0),
                        t_c: 0.0,
                        phase_offset: phase,
                        trail: Vec::new(),
                        slit_passed: 0,
                    });
                    total_fired += 1;
                }

            } else {
                // Analysis Complete - Calculate Metrics
                analyzing = false;
                paused = true; // Stay paused, don't return to interactive
                
                // 1. Extract Line Profile (Sum Z counts for each Y bin)
                let mut y_profile: HashMap<i32, u32> = HashMap::new();
                let mut min_y_idx = 1000;
                let mut max_y_idx = -1000;
                
                for ((gy, _), count) in &intensity_map {
                    *y_profile.entry(*gy).or_insert(0) += count;
                    min_y_idx = min_y_idx.min(*gy);
                    max_y_idx = max_y_idx.max(*gy);
                }
                
                // 2. Find Peaks and Minima
                let mut peaks: Vec<(f32, u32)> = Vec::new();
                let mut profile_vec: Vec<(f32, u32)> = Vec::new();
                let mut max_val = 0;
                let mut min_val = u32::MAX;
                
                // Prepare CSV content
                let mut csv_content = String::from("Bin_Y,Counts\n");

                for i in min_y_idx..=max_y_idx {
                    let val = *y_profile.get(&i).unwrap_or(&0);
                    let y_pos = i as f32 * GRID_SIZE;
                    profile_vec.push((y_pos, val));
                    max_val = max_val.max(val);
                    if val > 0 { min_val = min_val.min(val); }
                    
                    csv_content.push_str(&format!("{:.4},{}\n", y_pos, val));

                    // Simple local maxima check
                    if i > min_y_idx && i < max_y_idx {
                        let prev = *y_profile.get(&(i-1)).unwrap_or(&0);
                        let next = *y_profile.get(&(i+1)).unwrap_or(&0);
                        if val > prev && val > next && val > max_val / 4 { // Threshold to ignore noise
                            peaks.push((y_pos, val));
                        }
                    }
                }
                
                // Write CSV - use unique filename for sweeps
                let csv_filename = if sweep_mode {
                    format!("analysis_results_width_{:.1}.csv", current_slit_width)
                } else if phase_sweep_mode {
                    let phase_name = match phase_dist_mode { 0 => "uniform", 1 => "sinusoidal", 2 => "gaussian_pi", _ => "unknown" };
                    format!("analysis_results_phase_{}.csv", phase_name)
                } else {
                    "analysis_results.csv".to_string()
                };
                if let Err(e) = std::fs::write(&csv_filename, csv_content) {
                    println!("Failed to write CSV: {}", e);
                }
                
                // 3. Calculate Metrics
                // Fringe Spacing
                let mut spacing_sum = 0.0;
                let mut spacing_count = 0;
                if peaks.len() > 1 {
                    for i in 0..peaks.len()-1 {
                        spacing_sum += (peaks[i+1].0 - peaks[i].0).abs();
                        spacing_count += 1;
                    }
                }
                let mean_spacing = if spacing_count > 0 { spacing_sum / spacing_count as f32 } else { 0.0 };
                
                // Contrast Ratio (Michelson)
                let contrast = if max_val + min_val > 0 {
                    (max_val as f32 - min_val as f32) / (max_val as f32 + min_val as f32)
                } else { 0.0 };
                
                // FWHM (of central peak, assume closest to 0)
                let mut fwhm = 0.0;
                if let Some((center_peak_y, center_peak_val)) = peaks.iter().min_by(|a, b| a.0.abs().partial_cmp(&b.0.abs()).unwrap()) {
                    let half_max = *center_peak_val as f32 / 2.0;
                    // Find left crossing
                    let mut left_x = center_peak_y;
                    for (y, val) in profile_vec.iter().rev() {
                        if *y < *center_peak_y && *val as f32 <= half_max {
                            left_x = y;
                            break;
                        }
                    }
                    // Find right crossing
                    let mut right_x = center_peak_y;
                    for (y, val) in &profile_vec {
                        if *y > *center_peak_y && *val as f32 <= half_max {
                            right_x = y;
                            break;
                        }
                    }
                    fwhm = (right_x - left_x).abs();
                }

                // Symmetry (Correlation)
                // Simplified: compare sums of left vs right
                let left_sum: u32 = profile_vec.iter().filter(|(y, _)| *y < 0.0).map(|(_, v)| v).sum();
                let right_sum: u32 = profile_vec.iter().filter(|(y, _)| *y > 0.0).map(|(_, v)| v).sum();
                let symmetry = if left_sum + right_sum > 0 {
                    1.0 - (left_sum as f32 - right_sum as f32).abs() / (left_sum as f32 + right_sum as f32)
                } else { 1.0 };

                // Centroid and rotation metrics from the 2D hit map
                let mut sum_y = 0.0f32;
                let mut sum_z = 0.0f32;
                let mut sum_yy = 0.0f32;
                let mut sum_zz = 0.0f32;
                let mut sum_yz = 0.0f32;
                let mut total_weight = 0.0f32;
                for ((gy, gz), count) in &intensity_map {
                    let y = *gy as f32 * GRID_SIZE;
                    let z = *gz as f32 * GRID_SIZE;
                    let w = *count as f32;
                    sum_y += y * w;
                    sum_z += z * w;
                    sum_yy += y * y * w;
                    sum_zz += z * z * w;
                    sum_yz += y * z * w;
                    total_weight += w;
                }
                let centroid_y = if total_weight > 0.0 { sum_y / total_weight } else { 0.0 };
                let centroid_z = if total_weight > 0.0 { sum_z / total_weight } else { 0.0 };
                let var_y = if total_weight > 0.0 { sum_yy / total_weight - centroid_y * centroid_y } else { 0.0 };
                let var_z = if total_weight > 0.0 { sum_zz / total_weight - centroid_z * centroid_z } else { 0.0 };
                let cov_yz = if total_weight > 0.0 { sum_yz / total_weight - centroid_y * centroid_z } else { 0.0 };
                let pattern_angle_rad = 0.5 * (2.0 * cov_yz).atan2(var_y - var_z);
                let pattern_angle_deg = pattern_angle_rad * 180.0 / std::f32::consts::PI;
                let yz_corr = if var_y > 1e-6 && var_z > 1e-6 {
                    (cov_yz / (var_y * var_z).sqrt()).clamp(-1.0, 1.0)
                } else {
                    0.0
                };
                let slit_ratio = if slit2_hits > 0 {
                    slit1_hits as f32 / slit2_hits as f32
                } else {
                    slit1_hits as f32
                };

                metrics_text = format!(
                    "Angle: {:.1}°\nFired: {}\nHits: {}\nSpacing: {:.2}\nContrast: {:.2}\nFWHM: {:.2}\nSymmetry: {:.2}\nCentroidY: {:.2}\nCentroidZ: {:.2}\nPattern Tilt: {:.2}°\nYZ Corr: {:.2}\nSlit1: {}\nSlit2: {}\nMirrored: {}",
                    slit_angle, total_fired, analysis_hits, mean_spacing, contrast, fwhm, symmetry, centroid_y, centroid_z, pattern_angle_deg, yz_corr, slit1_hits, slit2_hits, mirrored
                );
                
                // Write Summary - use unique filename for sweeps
                let summary_filename = if sweep_mode {
                    format!("analysis_summary_width_{:.1}.txt", current_slit_width)
                } else if phase_sweep_mode {
                    let phase_name = match phase_dist_mode { 0 => "uniform", 1 => "sinusoidal", 2 => "gaussian_pi", _ => "unknown" };
                    format!("analysis_summary_phase_{}.txt", phase_name)
                } else {
                    "analysis_summary.txt".to_string()
                };
                if let Err(e) = std::fs::write(&summary_filename, &metrics_text) {
                    println!("Failed to write summary: {}", e);
                }
                if diagnostic_mode {
                    let mean_y = if diag_stats.count > 0 { diag_stats.sum_y / diag_stats.count as f32 } else { 0.0 };
                    let p_pos = if diag_stats.count > 0 { diag_stats.positive as f32 / diag_stats.count as f32 } else { 0.0 };
                    println!("Diagnostic symmetry -> samples: {}, mean_y: {:.4}, P(y>0): {:.4}", diag_stats.count, mean_y, p_pos);
                    metrics_text.push_str(&format!("\nDiag mean_y: {:.4}\nDiag P(y>0): {:.4}\nSamples: {}", mean_y, p_pos, diag_stats.count));
                    diagnostic_mode = false;
                }

                // Angle Sweep Mode: Store result and advance
                if angle_sweep_mode {
                    angle_sweep_results.push((slit_angle, centroid_y, centroid_z, pattern_angle_deg, yz_corr, contrast, symmetry, slit_ratio));
                    
                    angle_index += 1;
                    if angle_index < angle_sweep_values.len() {
                        // Next angle
                        slit_angle = angle_sweep_values[angle_index];
                        analysis_hits = 0;
                        total_fired = 0;
                        intensity_map.clear();
                        max_intensity = 1;
                        slit1_hits = 0;
                        slit2_hits = 0;
                        line_plot_points.clear();
                        analyzing = true;
                    } else {
                        // Angle sweep complete
                        angle_sweep_mode = false;
                        let mut csv = String::from("AngleDeg,CentroidY,CentroidZ,PatternAngleDeg,YZCorrelation,Contrast,Symmetry,SlitRatio\n");
                        for (a, cy, cz, pa, corr, c, s, sr) in &angle_sweep_results {
                            csv.push_str(&format!("{:.1},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}\n", a, cy, cz, pa, corr, c, s, sr));
                        }
                        if let Err(e) = std::fs::write("angle_sweep_results.csv", csv) {
                            println!("Failed to write angle sweep results: {}", e);
                        }
                        metrics_text = format!("Angle Sweep Complete!\n{} angles tested.\nResults in angle_sweep_results.csv", angle_sweep_results.len());
                        slit_angle = 0.0;
                    }
                }
                
                // Generate Line Plot Points for Visualization
                line_plot_points.clear();
                for (y, val) in profile_vec {
                    let scaled_val = (val as f32 / max_val as f32) * 10.0;
                    line_plot_points.push(vec2(y, scaled_val));
                }
                
                // Sweep Mode: Store result and advance
                if sweep_mode {
                    sweep_results.push((current_slit_width, fwhm, contrast, symmetry));
                    
                    // Advance to next width
                    current_slit_width += 0.2;
                    
                    if current_slit_width <= 2.01 { // Allow for float comparison
                        // Reset for next run
                        analysis_hits = 0;
                        intensity_map.clear();
                        max_intensity = 1;
                        slit1_hits = 0;
                        slit2_hits = 0;
                        analyzing = true; // Stay in analysis mode
                    } else {
                        // Sweep Complete - Write results
                        sweep_mode = false;
                        let mut csv = String::from("SlitWidth,FWHM,Contrast,Symmetry\n");
                        for (w, f, c, s) in &sweep_results {
                            csv.push_str(&format!("{:.2},{:.4},{:.4},{:.4}\n", w, f, c, s));
                        }
                        if let Err(e) = std::fs::write("sweep_results.csv", csv) {
                            println!("Failed to write sweep results: {}", e);
                        }
                        metrics_text = format!("Sweep Complete!\n{} widths tested.\nResults in sweep_results.csv", sweep_results.len());
                    }
                }
                
                // Phase Sweep Mode: Store result and advance
                if phase_sweep_mode {
                    let dist_name = match phase_dist_mode {
                        0 => "Uniform",
                        1 => "Sinusoidal",
                        2 => "Gaussian_Pi",
                        _ => "Unknown",
                    };
                    phase_sweep_results.push((dist_name.to_string(), contrast, symmetry, slit1_hits, slit2_hits));
                    
                    // Advance to next distribution
                    phase_dist_mode += 1;
                    
                    if phase_dist_mode <= 2 {
                        // Reset for next run
                        analysis_hits = 0;
                        total_fired = 0;
                        intensity_map.clear();
                        max_intensity = 1;
                        slit1_hits = 0;
                        slit2_hits = 0;
                        analyzing = true;
                    } else {
                        // Phase Sweep Complete - Write results
                        phase_sweep_mode = false;
                        phase_dist_mode = 0;
                        let mut csv = String::from("Distribution,Contrast,Symmetry,Slit1,Slit2\n");
                        for (name, c, s, s1, s2) in &phase_sweep_results {
                            csv.push_str(&format!("{},{:.4},{:.4},{},{}\n", name, c, s, s1, s2));
                        }
                        if let Err(e) = std::fs::write("phase_sweep_results.csv", csv) {
                            println!("Failed to write phase sweep results: {}", e);
                        }
                        metrics_text = format!("Phase Sweep Complete!\n3 distributions tested.\nResults in phase_sweep_results.csv");
                    }
                }
            }
        } else if !paused {
            // Normal interactive spawn (only if not paused)
            spawn_timer += 1.0;
            if particles.len() < 5000 {
                for _ in 0..20 {
                    let phase = rand::gen_range(0.0, 2.0 * std::f32::consts::PI);
                    let z0 = rand::gen_range(-Z_SPREAD, Z_SPREAD);
                    particles.push(Particle {
                        pos: vec3(0.0, 0.0, z0),
                        t_c: 0.0,
                        phase_offset: phase,
                        trail: Vec::new(),
                        slit_passed: 0,
                    });
                }

            }
        }

        // Update particles
        let mut i = 0;
        while i < particles.len() {
            let p = &mut particles[i];
            
            if p.t_c < TIME_LENGTH {
                let prev_x = p.pos.x;
                
                // Speed up physics during analysis
                let speed = if analyzing { 1.0 } else { PARTICLE_SPEED };
                p.t_c += speed;
                
                let x = p.t_c;
                // Quantum time stays as a phase-perturbed clock, but transverse motion uses a symmetric phase sum
                let t_q = p.t_c + LAMBDA * (OMEGA * p.t_c).sin();
                // Transverse position: symmetric in phase_offset; time modulation enters as an additive phase
                let mirror_sign = if mirrored { -1.0 } else { 1.0 };
                let y_phase = TRANSVERSE_FREQUENCY * t_q + p.phase_offset;
                
                let y = if wavefunction_mode {
                     mirror_sign * TRANSVERSE_AMPLITUDE * y_phase.cos()
                } else {
                     mirror_sign * TRANSVERSE_AMPLITUDE * y_phase.sin()
                };
                
                let z = if wavefunction_mode {
                    p.pos.z + TRANSVERSE_AMPLITUDE * y_phase.sin() // Add imaginary part to Z
                } else {
                    p.pos.z // Preserve initial z
                };
                
                p.pos = vec3(x, y, z);
                
                if !analyzing {
                    p.trail.push(p.pos);
                    if p.trail.len() > 50 {
                        p.trail.remove(0);
                    }
                }

                // Double Slit Check
                if prev_x < barrier_x && x >= barrier_x {
                    if diagnostic_mode {
                        // Collect symmetry diagnostics and discard particle
                        diag_stats.count += 1;
                        diag_stats.sum_y += y;
                        if y > 0.0 { diag_stats.positive += 1; }
                        particles.remove(i);
                        continue;
                    }
                    if slit_enabled {
                        // Gaussian Slit Transmission with angle tilt
                        // When slits are tilted, the center shifts based on z position
                        // For simplicity, we shift centers based on the angle
                        // tan(angle) = shift_y / reference_distance
                        
                        let angle_rad = slit_angle * std::f32::consts::PI / 180.0;
                        let angle_shift = z * angle_rad.tan(); // Shift based on z and angle
                        
                        let sigma = 0.4; // Tunable for "softness"
                        let center_1 = -s2 + angle_shift;
                        let center_2 = s2 + angle_shift;
                        
                        let p1 = (-((y - center_1).powi(2)) / (2.0 * sigma * sigma)).exp();
                        let p2 = (-((y - center_2).powi(2)) / (2.0 * sigma * sigma)).exp();
                        
                        let transmission_prob = p1 + p2;
                        
                        if rand::gen_range(0.0, 1.0) > transmission_prob {
                            // Blocked/Absorbed
                            particles.remove(i);
                            continue;
                        }
                        
                        // Track which slit: compare distances
                        if (y - center_1).abs() < (y - center_2).abs() {
                            p.slit_passed = 1; // Slit 1 (negative Y)
                        } else {
                            p.slit_passed = 2; // Slit 2 (positive Y)
                        }
                    }
                }
                
                i += 1;
            } else {
                // Reached Screen
                let key = grid_key(p.pos.y, p.pos.z, GRID_SIZE);
                let count = intensity_map.entry(key).or_insert(0);
                *count += 1;
                max_intensity = max_intensity.max(*count);
                
                if analyzing {
                    analysis_hits += 1;
                    if p.slit_passed == 1 {
                        slit1_hits += 1;
                    } else if p.slit_passed == 2 {
                        slit2_hits += 1;
                    }
                }
                
                particles.remove(i);

            }
        }

        // Draw active particles and trails (Only if not analyzing for speed)
        if !analyzing {
            for p in &particles {
                draw_sphere(p.pos, 0.15, None, Color::new(1.0, 1.0, 0.0, 0.8));
                
                if p.trail.len() > 1 {
                    for j in 0..p.trail.len() - 1 {
                        let color = Color::new(0.5, 0.5, 1.0, 0.4);
                        draw_line_3d(p.trail[j], p.trail[j+1], color);
                    }
                }
            }
        }
        
        // Draw intensity map on Screen (X = TIME_LENGTH)
        for ((gy, gz), count) in &intensity_map {
            let y = *gy as f32 * GRID_SIZE;
            let z = *gz as f32 * GRID_SIZE;
            
            let intensity = (*count as f32 / max_intensity as f32).sqrt(); 
            
            let color = if intensity < 0.2 {
                Color::new(0.0, 0.0, intensity * 5.0, 0.8)
            } else if intensity < 0.4 {
                Color::new(0.0, (intensity - 0.2) * 5.0, 1.0, 0.8)
            } else if intensity < 0.6 {
                Color::new((intensity - 0.4) * 5.0, 1.0, 0.0, 0.8)
            } else if intensity < 0.8 {
                Color::new(1.0, 1.0 - (intensity - 0.6) * 5.0, 0.0, 0.8)
            } else {
                Color::new(1.0, 1.0, (intensity - 0.8) * 5.0, 0.8)
            };
            
            draw_cube(vec3(TIME_LENGTH, y, z), vec3(0.1, GRID_SIZE, GRID_SIZE), None, color);
        }
        
        // Draw Line Plot Overlay
        if !line_plot_points.is_empty() {
            for i in 0..line_plot_points.len()-1 {
                let p1 = line_plot_points[i];
                let p2 = line_plot_points[i+1];
                // Draw floating above the screen
                draw_line_3d(
                    vec3(TIME_LENGTH - p1.y, p1.x, 5.0), // Rotate: Y->X, Val->Z (height)
                    vec3(TIME_LENGTH - p2.y, p2.x, 5.0),
                    WHITE
                );
            }
        }

        // Reset
        if is_key_pressed(KeyCode::Space) {
            particles.clear();
            intensity_map.clear();
            max_intensity = 1;
            analysis_hits = 0;
            metrics_text.clear();
            line_plot_points.clear();
            analyzing = false;
            paused = false; // Resume interactive mode
            slit1_hits = 0;
            slit2_hits = 0;
            sweep_mode = false;
            phase_sweep_mode = false;
            angle_sweep_mode = false;
            diagnostic_mode = false;
            current_slit_width = 0.2;
            sweep_results.clear();
            angle_sweep_results.clear();
            angle_index = 0;
            diag_stats = DiagnosticStats { count: 0, sum_y: 0.0, positive: 0 };
        }
        
        // Start Analysis
        if is_key_pressed(KeyCode::A) {
            analyzing = true;
            angle_sweep_mode = false;
            particles.clear();
            intensity_map.clear();
            max_intensity = 1;
            analysis_hits = 0;
            total_fired = 0;
            metrics_text.clear();
            line_plot_points.clear();
            slit1_hits = 0;
            slit2_hits = 0;
            angle_sweep_results.clear();
        }
        
        // Toggle Mirror
        if is_key_pressed(KeyCode::M) {
            mirrored = !mirrored;
        }

        // Toggle Slit
        if is_key_pressed(KeyCode::E) {
            slit_enabled = !slit_enabled;
        }

        // Toggle Wavefunction Mode
        if is_key_pressed(KeyCode::W) {
            wavefunction_mode = !wavefunction_mode;
        }
        
        // Start Sweep
        if is_key_pressed(KeyCode::S) {
            sweep_mode = true;
            angle_sweep_mode = false;
            diagnostic_mode = false;
            analyzing = true;
            current_slit_width = 0.2;
            sweep_results.clear();
            angle_sweep_results.clear();
            angle_index = 0;
            particles.clear();
            intensity_map.clear();
            max_intensity = 1;
            analysis_hits = 0;
            metrics_text.clear();
            line_plot_points.clear();
            slit1_hits = 0;
            slit2_hits = 0;
            total_fired = 0;
        }
        
        // Start Phase Sweep
        if is_key_pressed(KeyCode::P) {
            phase_sweep_mode = true;
            angle_sweep_mode = false;
            diagnostic_mode = false;
            analyzing = true;
            phase_dist_mode = 0;
            phase_sweep_results.clear();
            angle_sweep_results.clear();
            angle_index = 0;
            particles.clear();
            intensity_map.clear();
            max_intensity = 1;
            analysis_hits = 0;
            metrics_text.clear();
            line_plot_points.clear();
            slit1_hits = 0;
            slit2_hits = 0;
            total_fired = 0;
        }

        // Start Angle/Tilt Sweep
        if is_key_pressed(KeyCode::T) {
            angle_sweep_mode = true;
            sweep_mode = false;
            phase_sweep_mode = false;
            diagnostic_mode = false;
            analyzing = true;
            angle_index = 0;
            slit_angle = angle_sweep_values[angle_index];
            angle_sweep_results.clear();
            particles.clear();
            intensity_map.clear();
            max_intensity = 1;
            analysis_hits = 0;
            metrics_text.clear();
            line_plot_points.clear();
            slit1_hits = 0;
            slit2_hits = 0;
            total_fired = 0;
        }

        // Diagnostic symmetry check (no slit transmission, stop at barrier)
        if is_key_pressed(KeyCode::D) {
            diagnostic_mode = true;
            sweep_mode = false;
            phase_sweep_mode = false;
            angle_sweep_mode = false;
            analyzing = true;
            particles.clear();
            intensity_map.clear();
            max_intensity = 1;
            analysis_hits = 0;
            total_fired = 0;
            metrics_text.clear();
            line_plot_points.clear();
            slit1_hits = 0;
            slit2_hits = 0;
            diag_stats = DiagnosticStats { count: 0, sum_y: 0.0, positive: 0 };
            slit_angle = 0.0;
        }

        set_default_camera();
        
        draw_text("Quantum Time Simulation (Analysis Mode)", 10.0, 30.0, 20.0, WHITE);
        let phase_name = match phase_dist_mode { 0 => "Uniform", 1 => "Sinusoidal", 2 => "Gaussian(π)", _ => "?" };
        let status_text = if phase_sweep_mode {
            format!("PHASE SWEEP: {} | Fired: {} | Hits: {} | Step {}/3", phase_name, total_fired, analysis_hits, phase_sweep_results.len() + 1)
        } else if angle_sweep_mode {
            format!("ANGLE SWEEP: {:.1}° | Fired: {} | Hits: {} | Step {}/{}", slit_angle, total_fired, analysis_hits, angle_sweep_results.len() + 1, angle_sweep_values.len())
        } else if diagnostic_mode {
            format!("DIAGNOSTIC: Fired: {} | Samples: {}", total_fired, diag_stats.count)
        } else if sweep_mode {
            format!("SWEEP: Width={:.1} | Fired: {} | Hits: {} | Step {}/10", current_slit_width, total_fired, analysis_hits, sweep_results.len() + 1)
        } else {
            format!("Fired: {} | Hits: {} | Phase: {} | Mirror: {} | Slit: {} | Wave: {} | Angle: {:.1}°", total_fired, if analyzing { analysis_hits } else { intensity_map.len() }, phase_name, if mirrored { "ON" } else { "OFF" }, if slit_enabled { "ON" } else { "OFF" }, if wavefunction_mode { "ON" } else { "OFF" }, slit_angle)
        };
        draw_text(&status_text, 10.0, 50.0, 20.0, YELLOW);
        draw_text("SPACE reset | A Analyze | M Mirror | E Slit | W Wave | S Sweep | P Phase | T Tilt | D Diag", 10.0, 70.0, 20.0, LIGHTGRAY);
        draw_text(&format!("Slit1: {} | Slit2: {}", slit1_hits, slit2_hits), 10.0, 90.0, 20.0, ORANGE);
        
        // Draw Metrics
        if !metrics_text.is_empty() {
            let lines: Vec<&str> = metrics_text.split('\n').collect();
            for (i, line) in lines.iter().enumerate() {
                draw_text(line, 10.0, 110.0 + i as f32 * 20.0, 20.0, GREEN);
            }
        }

        next_frame().await
    }
}
