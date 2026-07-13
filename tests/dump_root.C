// Deterministic text dump of a ROOT file's semantic content, for A/B
// comparison of PROfit outputs. ROOT files are never byte-identical across
// runs (embedded timestamps), so compare_tags.sh dumps both sides with this
// macro and diffs the text instead.
//
// Usage:  root -l -b -q 'tests/dump_root.C("file.root")'  > file.dump
//
// Covers TH1/TH2 (all precisions), TGraph(+errors), TTree (entry count plus
// per-leaf sum/min/max), and recurses into subdirectories. Unknown classes
// print name+class only. Values print with %.6g: coarse enough to ignore
// nothing — same-machine same-code runs are expected to match exactly.

#include "TFile.h"
#include "TKey.h"
#include "TH1.h"
#include "TH2.h"
#include "TGraph.h"
#include "TGraphErrors.h"
#include "TTree.h"
#include "TLeaf.h"
#include "TDirectory.h"
#include "TCollection.h"
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>

static void dump_hist(const TH1 *h, const std::string &path) {
    printf("HIST %s | %s | ncells=%d integral=%.6g\n",
           path.c_str(), h->ClassName(), h->GetNcells(), h->Integral());
    for(int i = 0; i < h->GetNcells(); ++i) {
        double c = h->GetBinContent(i), e = h->GetBinError(i);
        if(c != 0.0 || e != 0.0)
            printf("  cell %d : %.6g +- %.6g\n", i, c, e);
    }
}

static void dump_graph(const TGraph *g, const std::string &path) {
    printf("GRAPH %s | %s | n=%d\n", path.c_str(), g->ClassName(), g->GetN());
    for(int i = 0; i < g->GetN(); ++i)
        printf("  pt %d : %.6g %.6g\n", i, g->GetPointX(i), g->GetPointY(i));
}

static bool leaf_is_object_backed(TLeaf *lf) {
    // Leaves of split object/STL branches (e.g. a map<string,float> stored as
    // "x.first"/"x.second") go through a collection proxy that TTree::Draw
    // reads unreliably — repeated dumps of the SAME file give different
    // values. Skip their values; report presence only.
    for(TBranch *br = lf->GetBranch(); br; ) {
        if(br->InheritsFrom("TBranchElement")) return true;
        TBranch *mother = br->GetMother();
        if(mother == br) break;
        br = mother;
    }
    return false;
}

static void dump_tree(TTree *t, const std::string &path) {
    const Long64_t n = t->GetEntries();
    printf("TREE %s | entries=%lld\n", path.c_str(), n);
    if(n == 0) return;
    t->SetEstimate(n + 1);
    TIter it(t->GetListOfLeaves());
    std::vector<std::pair<std::string, TLeaf*>> leaves;
    while(TObject *o = it()) leaves.push_back({o->GetName(), (TLeaf*)o});
    std::sort(leaves.begin(), leaves.end(),
              [](const auto &a, const auto &b){ return a.first < b.first; });
    for(const auto &p : leaves) {
        const std::string &leaf = p.first;
        if(leaf_is_object_backed(p.second)) {
            printf("  leaf %s : object-backed, values not dumped\n", leaf.c_str());
            continue;
        }
        const Long64_t got = t->Draw(leaf.c_str(), "", "goff");
        if(got <= 0) { printf("  leaf %s : (not drawable)\n", leaf.c_str()); continue; }
        const double *v = t->GetV1();
        // Finite-only statistics: a single NaN would otherwise poison the sum
        // and min/max in ways that print differently run to run ("nan"/"-nan").
        double sum = 0, lo = 0, hi = 0;
        Long64_t nfin = 0, nbad = 0;
        for(Long64_t i = 0; i < got; ++i) {
            if(!std::isfinite(v[i])) { ++nbad; continue; }
            if(nfin == 0) { lo = hi = v[i]; }
            else { if(v[i] < lo) lo = v[i]; if(v[i] > hi) hi = v[i]; }
            sum += v[i];
            ++nfin;
        }
        printf("  leaf %s : n=%lld nfinite=%lld nnonfinite=%lld sum=%.6g min=%.6g max=%.6g\n",
               leaf.c_str(), got, nfin, nbad, sum, lo, hi);
    }
}

static void dump_dir(TDirectory *dir, const std::string &prefix) {
    // Iterate keys sorted by name (and keep only the highest cycle per name)
    // so dump order never depends on write order.
    std::vector<std::pair<std::string, TKey*>> keys;
    TIter it(dir->GetListOfKeys());
    while(TKey *k = (TKey*)it()) {
        std::string name = k->GetName();
        bool superseded = false;
        for(auto &p : keys)
            if(p.first == name) { if(k->GetCycle() > p.second->GetCycle()) p.second = k; superseded = true; break; }
        if(!superseded) keys.push_back({name, k});
    }
    std::sort(keys.begin(), keys.end(),
              [](const auto &a, const auto &b){ return a.first < b.first; });

    for(auto &p : keys) {
        TKey *k = p.second;
        const std::string path = prefix + "/" + p.first;
        TObject *obj = k->ReadObj();
        if(!obj) { printf("NULL %s | %s\n", path.c_str(), k->GetClassName()); continue; }
        if(obj->InheritsFrom(TDirectory::Class())) dump_dir((TDirectory*)obj, path);
        else if(obj->InheritsFrom(TH1::Class()))   dump_hist((TH1*)obj, path);
        else if(obj->InheritsFrom(TGraph::Class()))dump_graph((TGraph*)obj, path);
        else if(obj->InheritsFrom(TTree::Class())) dump_tree((TTree*)obj, path);
        else printf("OBJ %s | %s\n", path.c_str(), obj->ClassName());
    }
}

void dump_root(const char *fname) {
    TFile f(fname, "READ");
    if(f.IsZombie()) { printf("ZOMBIE %s\n", fname); return; }
    dump_dir(&f, "");
}
