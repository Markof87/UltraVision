#include "classification.h"

std::string classifyDynamicAction(const SequenceFeatures& feats) {
    
    // 1. RUNNING
    // Velocity > 6.4 captures pure running.
    // Velocity > 4.3 combined with long strides (<= 3 peaks) isolates slower runs 
    // from fast jogging (which has >= 4 peaks at high speeds).
    if (feats.avg_velocity > 6.4f || (feats.avg_velocity > 4.3f && feats.stride_peaks <= 3)) {
        return "running";
    }
    
    // 2. JOGGING
    // A velocity threshold of 3.0 provides the mathematically optimal split 
    // between normal walking and jogging energy levels.
    if (feats.avg_velocity > 3.0f) {
        return "jogging";
    }
    
    // 3. WALKING
    // Default class for slow, low-energy movements.
    return "walking";
}

std::string classifyStaticAction(const StaticFeatures& feats) {

    // 1. BOXING
    // Lateral asymmetry > 0.074 captures boxing punches.
    // Forward arm extension moves the motion centroid away from the body midline,
    // distinguishing it from the centered, two-handed clapping and waving actions.
    if (feats.avg_x_offset > 0.074) {
        return "boxing";
    }

    // 2. HANDWAVING
    // Highest vertical reach min_cy < 0.19 combined with centered symmetry (avg_x_offset < 0.055)
    // captures handwaving. The hands periodically travel above head level (low normalized Y)
    // while remaining horizontally balanced across the bounding box.
    else if (feats.min_cy < 0.19 && feats.avg_x_offset < 0.055) {
        return "handwaving";
    }

    // 3. HANDCLAPPING
    // Default class for stationary, centered movements that remain confined around chest height.
    else {
        return "handclapping";
    }
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