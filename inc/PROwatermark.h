#ifndef PROWATERMARK_H
#define PROWATERMARK_H

// PROfit version watermark. Every PDF/PNG page PROfit writes should carry the
// version string; call drawVersionWatermark on the TOP-LEVEL canvas (not a
// sub-pad) after the page content is drawn, immediately before Print/SaveAs.
// Multi-page loops that Clear() the canvas must re-stamp each page; a canvas
// printed to several formats back-to-back needs only one stamp.

#include <algorithm>
#include <string>

#include "TH2.h"
#include "TPad.h"
#include "TText.h"
#include "TVirtualPad.h"

#include "PROversion.h"

namespace PROfit {

    enum class WatermarkPos {
        TopRight,    // default: bottom-right of the text sits just above the frame's top-right corner
        BottomRight, // side-by-side panel pages where top-right NDC lands inside a sub-pad
        RightEdge    // dense Divide() grids: no free corner, so run vertically up the right-edge strip
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
            // Anchor the text's bottom-right just above the top-right corner of the axis
            // frame. Default to the pad's own margins; if sub-pads are already drawn
            // (stacked main+ratio layouts), align to the top-most one's frame instead.
            double x = 1.0 - pad->GetRightMargin();
            double y = 1.0 - pad->GetTopMargin();
            double best_top = -1.0;
            TVirtualPad *frame_pad = pad;
            TIter next(pad->GetListOfPrimitives());
            while(TObject *o = next()) {
                if(auto *sub = dynamic_cast<TPad*>(o)) {
                    const double top = sub->GetYlowNDC() + sub->GetHNDC();
                    if(top > best_top) {
                        best_top = top;
                        frame_pad = sub;
                        x = sub->GetXlowNDC() + (1.0 - sub->GetRightMargin()) * sub->GetWNDC();
                        y = sub->GetYlowNDC() + (1.0 - sub->GetTopMargin()) * sub->GetHNDC();
                    }
                }
            }
            // colz pages draw the z-palette's exponent (x10^n) just above the frame's
            // top-right corner; lift the stamp clear of it when the frame holds a TH2.
            bool has_th2 = false;
            TIter pit(frame_pad->GetListOfPrimitives());
            while(TObject *o = pit()) if(dynamic_cast<TH2*>(o)) { has_th2 = true; break; }
            const double gap = has_th2 ? 0.024 : 0.004;
            t.SetTextSize(0.032f);
            t.SetTextAlign(31); // right-bottom anchor
            t.DrawText(x, std::min(y + gap, 0.97), pv.c_str()); // DrawText clones; the pad owns the clone
        } else if(pos == WatermarkPos::BottomRight) {
            t.SetTextSize(0.025f);
            t.SetTextAlign(31); // right-bottom anchor
            t.DrawText(0.995, 0.005, pv.c_str());
        } else { // RightEdge
            // Full Divide() grids own both corners (bottom-right cell's x title, top
            // row's panel titles); the right column's pad margins leave an empty strip
            // along the canvas edge — run the stamp vertically up it.
            t.SetTextSize(0.022f);
            t.SetTextAngle(90);
            t.SetTextAlign(11); // rotated: bottom-left anchor, text runs upward
            t.DrawText(0.998, 0.005, pv.c_str());
        }
        if(prev) prev->cd();
    }

}

#endif
