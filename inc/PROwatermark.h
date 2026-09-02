#ifndef PROWATERMARK_H
#define PROWATERMARK_H

// PROfit version watermark. Every PDF/PNG page PROfit writes should carry the
// version string; call drawVersionWatermark on the TOP-LEVEL canvas (not a
// sub-pad) after the page content is drawn, immediately before Print/SaveAs.
// Multi-page loops that Clear() the canvas must re-stamp each page; a canvas
// printed to several formats back-to-back needs only one stamp.

#include <string>

#include "TText.h"
#include "TVirtualPad.h"

#include "PROversion.h"

namespace PROfit {

    enum class WatermarkPos {
        TopRight,    // default: in the canvas top margin, above the frame
        BottomRight  // multi-panel pages where top-right NDC lands inside a sub-pad
    };

    inline void drawVersionWatermark(TVirtualPad *pad, WatermarkPos pos = WatermarkPos::TopRight) {
        if(!pad) return;
        TVirtualPad *prev = gPad;
        pad->cd();
        TText t;
        t.SetNDC();
        t.SetTextFont(42);
        const std::string pv = "PROfit v" + std::string(PROJECT_VERSION_STR);
        if(pos == WatermarkPos::TopRight) {
            t.SetTextSize(0.028f);
            t.SetTextAlign(33); // right-top anchor
            t.DrawText(0.96, 0.985, pv.c_str()); // DrawText clones; the pad owns the clone
        } else {
            t.SetTextSize(0.022f);
            t.SetTextAlign(31); // right-bottom anchor
            t.DrawText(0.995, 0.005, pv.c_str());
        }
        if(prev) prev->cd();
    }

}

#endif
