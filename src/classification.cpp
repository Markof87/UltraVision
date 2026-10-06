#include "classification.h"

std::string classifyStaticAction(const StaticFeatures& feats) {

    // -------------------------------------------------------------------------
    // 1. BOXING
    // -------------------------------------------------------------------------
    // Rule A: feats.avg_x_offset > 0.078
    //   - feats.avg_x_offset measures the horizontal distance of the motion centroid 
    //     from the vertical body centerline, normalized by the bbox width (|cx - center_x| / width).
    //   - In forward punches, the arm extends to one side, pushing avg_x_offset > 0.078.
    //   - Handclapping is centered and almost always stays <= 0.078.
    //
    // Rule B: feats.avg_x_offset > 0.0665 && feats.y_range > 0.410
    //   - feats.y_range is the total vertical span of the motion centroid (max_cy - min_cy) normalized by bbox height.
    //   - In outdoor d2 sequences, distance compresses horizontal displacement (down to 0.067 - 0.069),
    //     but the full punch-and-guard cycle spans a large vertical interval (feats.y_range > 0.410).
    //   - Clapping with similar lateral drift has a much smaller vertical span (feats.y_range <= 0.334).
    if (feats.avg_x_offset > 0.078 || 
       (feats.avg_x_offset > 0.0665 && feats.y_range > 0.410)) 
    {
        return "boxing";
    }

    // -------------------------------------------------------------------------
    // 2. HANDWAVING
    // -------------------------------------------------------------------------
    // Rule A: feats.min_cy < 0.185 && feats.avg_x_offset <= 0.055
    //   - feats.min_cy is the highest vertical position of the motion centroid 
    //     (0.0 = top of head, 1.0 = feet).
    //   - Arms raised above shoulders/head pull feats.min_cy below 0.185.
    //   - feats.avg_x_offset <= 0.055 ensures the movement is two-handed and symmetric 
    //     (distinguishing it from high punches).
    //
    // Rule B: feats.min_cy < 0.285 && feats.y_range > 0.310 && feats.avg_x_offset < 0.050
    //   - Recovers waving sequences (like person03_d3) where segmentation doesn't fully reach the head 
    //     (min_cy sits at 0.279), but large cyclic arm swings still create wide vertical motion (feats.y_range > 0.310) 
    //     with strict horizontal symmetry (feats.avg_x_offset < 0.050).
    else if ((feats.min_cy < 0.185 && feats.avg_x_offset <= 0.055) ||
             (feats.min_cy < 0.285 && feats.y_range > 0.310 && feats.avg_x_offset < 0.050)) 
    {
        return "handwaving";
    }

    // -------------------------------------------------------------------------
    // 3. HANDCLAPPING
    // -------------------------------------------------------------------------
    // Default fallback: stationary motion strictly centered on the torso (feats.avg_x_offset <= 0.078)
    // with motion confined to chest height (feats.min_cy >= 0.185).
    else {
        return "handclapping";
    }
}


std::string classifyDynamicAction(const SequenceFeatures& feats) {
    
    // -------------------------------------------------------------------------
    // 1. RUNNING
    // -------------------------------------------------------------------------
    // Rule A: feats.avg_velocity > 5.8
    //   - feats.avg_velocity is the mean displacement of the bbox center in pixels/frame.
    //   - 5.8 is the highest velocity recorded by any jogging sequence (person03_d3 was 5.8).
    //     Anything strictly above 5.8 is definitely running.
    //
    // Rule B: feats.avg_velocity > 3.5 && feats.stride_peaks <= 3 && feats.net_displacement < 140.0
    //   - feats.stride_peaks counts how many times the bbox width oscillates due to legs opening/closing.
    //   - feats.net_displacement is the Euclidean distance between the first and last frame centroids.
    //   - Running with a long stride produces few cycles (stride_peaks <= 3) at moderate speed (> 3.5).
    //   - Setting feats.net_displacement < 140.0 isolates these runs from fast jogging sequences 
    //     (like person01_d3 and person02_d1) which also have 3 peaks but cover longer trajectories (>= 144.0).
    if (feats.avg_velocity > 5.8f || 
       (feats.avg_velocity > 3.5f && feats.stride_peaks <= 3 && feats.net_displacement < 140.0f)) 
    {
        return "running";
    }
    
    // -------------------------------------------------------------------------
    // 2. JOGGING
    // -------------------------------------------------------------------------
    // Rule A: feats.avg_velocity > 2.4 && feats.stride_peaks >= 4
    //   - Captures typical jogging cadence: small rapid steps create frequent width expansions 
    //     (stride_peaks >= 4) even at slower speeds.
    //
    // Rule B: feats.avg_velocity > 3.2
    //   - Captures faster jogging. Normal walking almost always stays at or below 2.8 pixels/frame,
    //     making 3.2 a reliable speed threshold to separate jogging from walking.
    if ((feats.avg_velocity > 2.4f && feats.stride_peaks >= 4) || feats.avg_velocity > 3.2f) {
        return "jogging";
    }
    
    // -------------------------------------------------------------------------
    // 3. WALKING
    // -------------------------------------------------------------------------
    // Default fallback: captures sequences with low velocity (feats.avg_velocity <= 3.2) 
    // and low stride frequency (feats.stride_peaks < 4).
    return "walking";
}


std::string classifySequence(const SequenceFeatures& dyn_feats,
                                              const std::vector<cv::Mat>& mask_sequence,
                                              const std::vector<cv::Rect>& bbox_sequence) 
{
    // Macro router: separate dynamic and static actions
    bool is_dynamic = (dyn_feats.net_displacement > 25.0f || dyn_feats.avg_velocity > 4.0f);

    if (is_dynamic) {
        return classifyDynamicAction(dyn_feats);
    } else {
        StaticFeatures static_feats = computeStaticFeatures(mask_sequence, bbox_sequence);
        return classifyStaticAction(static_feats);
    }
}