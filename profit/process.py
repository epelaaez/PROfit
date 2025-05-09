import profit
import numpy as np
import pandas as pd
import uproot
import awkward as ak
import time

clock = time.perf_counter

# Takes a config object, load SystStruct's and PROpeller through dataframe interface
def PROcess_dataframes(c):
    # open up the files
    t1 = clock()
    ttree_dfs = []
    for fname, tree in zip(c.m_mcgen_file_name, c.m_mcgen_tree_name):
        ttree_dfs.append(load_df(fname, tree))
    t2 = clock()

    profit.PROlogINFO("DONE OPENING FILES [%f seconds]" % (t2-t1))

    # load systematic weights
    t1 = clock()
    eventweights = {}
    for i_f, (f, ttree) in enumerate(zip(c.m_mcgen_file_name, ttree_dfs)):
         if f not in eventweights:
             eventweights[f] = loadsysts(f, ttree, c)
         compute_branches(f, i_f, ttree, c)

    # List of systematics
    syst_structs = init_syst_structs(c, next(iter(eventweights.values())))
    t2 = clock()

    profit.PROlogINFO("DONE LOADING SYSTEMATICS [%f seconds]" % (t2-t1))

    # Process events
    t1 = clock()
    prop = init_propeller(c)

    for idf, tdf in enumerate(ttree_dfs):
        process_events(tdf, idf, syst_structs, eventweights[c.m_mcgen_file_name[idf]], prop, c)
    t2 = clock()

    profit.PROlogINFO("DONE PROCESSING EVENTS [%f seconds]" % (t2-t1))

    return syst_structs, prop

# Various helper functions

def init_syst_structs(c, ew):
    syst_structs = []

    # map_systematic_num_universe <- number of universes for each systematic
    map_systematic_num_universe = {}
    systematics = ew.systematics()
    for s in c.m_mcgen_variation_allowlist:
         if s in systematics:
             map_systematic_num_universe[s] = ew.nvariation(s)

    total_num_systematics = len(map_systematic_num_universe)
    for syst, nuniv in map_systematic_num_universe.items():
        mode = c.m_mcgen_variation_type_map[syst]

        syst_structs.append(profit.SystStruct(syst, nuniv))
        syst_structs[-1].SetWeightFormula("1")
        syst_structs[-1].SetMode(mode)

        if mode == "spline":
            syst_structs[-1].knobval = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32)
            syst_structs[-1].knob_index = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32)
            syst_structs[-1].binning = c.m_mcgen_variation_binning_map[syst]

    for s in syst_structs:
        s.SanityCheck()

    for s in syst_structs:
        if s.mode == "covariance":
            nbins = c.m_num_bins_total
        else:
            if s.binning == -2:
                nbins = c.m_num_truebins_total
            elif s.binning == -1:
                nbins = c.m_num_bins_total
            else:
                pass # TODO: implement other binning

        s.CreateSpecs(nbins)

    return syst_structs

def init_propeller(c):
    prop = profit.PROpeller()
    prop.hist = np.zeros((c.m_num_truebins_total, c.m_num_bins_total))
    prop.histLE = np.zeros((c.m_num_truebins_total,))
    prop.mcStatErr = np.zeros((c.m_num_truebins_total,))

    edges = np.array(c.m_channel_truebin_edges)
    edges_lo = edges[:, :-1].flatten()
    edges_hi = edges[:, 1:].flatten()
    # TODO: WHY IS THIS THE WRONG SIZE???
    prop.histLE[:edges_lo.size] += (edges_lo + edges_hi) / 2.

    return prop

def load_df(fname, treename, branches=None, allowed_branches=None, concat="outer"):
    if fname.endswith(".root"):
        tf = uproot.open(fname)
        if allowed_branches is not None and branches is None:
            branches = [b for b in tf[treename].keys() if b in allowed_branches]

        df = make_array(tf[treename], branches, concat=concat)
    elif fname.endswith(".df"):
        df = pd.read_hdf(fname, key=treename)
        if branches is not None:
            df = df[branches]
    else: 
        print(fname)
        assert(False)

    # if concat is None, we should return a list
    if concat is None and not isinstance(df, list):
        return [df]

    return df

def make_array(ttree, branches=None, concat="outer"):
    if branches is None:
        branches = ttree.keys()

    return ak.to_dataframe(ttree.arrays(branches, library="ak"), how=concat)

def wgtname(c):
    if isinstance(c, tuple):
        return c[0]
    return c

def systematics_df(df, c):
    # Restrict to the configured systematics
    df = df[[k for k in df.columns if wgtname(k) in c.m_mcgen_variation_allowlist and wgtname(k) not in c.m_mcgen_variation_denylist]]
    return profit.SystematicsDF.build(df, c.m_mcgen_variation_type_map) 

def compute_branches(fname, fid, ttree_df, c):
    for ib in range(len(c.m_branch_variables[fid])):
        branch = c.m_branch_variables[fid][ib]
        c.m_branch_variables[fid][ib].branch_formula = profit.DataFrameFormula("branch_form_%i_%i" % (fid, ib), branch.name, ttree_df)
        c.m_branch_variables[fid][ib].branch_true_L_formula = profit.DataFrameFormula("branch_L_form_%i_%i" % (fid, ib), branch.true_L_name, ttree_df)
        c.m_branch_variables[fid][ib].branch_true_value_formula = profit.DataFrameFormula("branch_true_form_%i_%i" % (fid, ib), branch.true_param_name, ttree_df)
        c.m_branch_variables[fid][ib].branch_true_pdg_formula = profit.DataFrameFormula("branch_pdg_form_%i_%i" % (fid, ib), branch.pdg_name, ttree_df)

        if c.m_mcgen_additional_weight_bool[fid][ib]:
            c.m_branch_variables[fid][ib].branch_monte_carlo_weight_formula = profit.DataFrameFormula("branch_add_weight_%i_%i" % (fid, ib), c.m_mcgen_additional_weight_name[fid][ib], ttree_df)

def loadsysts(fname, ttree_df, c):
    friends = []
    friend_names = []
    for friend, friend_f in zip(c.m_mcgen_file_friend_treename_map.get(fname, []), c.m_mcgen_file_friend_map.get(fname, [])):
        # Remove duplicates (TODO: why are they there???)
        if friend in friend_names: continue

        friend_names.append(friend)
        for df in load_df(friend_f, friend, concat=None, allowed_branches=c.m_mcgen_variation_allowlist):
            friends.append(df)

    # get the event weights
    event_weights_pertree = [systematics_df(df, c) for df in [ttree_df] + friends]
    event_weights_pertree = profit.SystematicsDF.concat([df for df in event_weights_pertree if not df.empty], axis=1)

    return event_weights_pertree

def process_branch(c, branch, evws, mcpot, subchannel_index, syst_vector, syst_additional_weight, inprop):
    # Load values
    total_num_sys = len(syst_vector)
    reco_value = branch.GetValue()
    true_param = branch.GetTrueValue()

    baseline = branch.GetTrueL()
    true_value = baseline / true_param
    pdg_id = branch.GetTruePDG()
    run_syst = branch.GetIncludeSystematics()
    mc_weight = branch.GetMonteCarloWeight()
    mc_weight *= c.m_plot_pot / mcpot

    if not isinstance(mc_weight, pd.Series):
        mc_weight = pd.Series(mc_weight, true_param.index)

    model_rule = pd.Series(branch.GetModelRule(), true_param.index)
    run_syst = branch.GetIncludeSystematics()

    global_bin = profit.FindGlobalBin(c, reco_value, subchannel_index)
    global_true_bin = profit.FindGlobalTrueBin(c, true_value, subchannel_index) if run_syst else pd.Series(0, true_param.index)

    valid = (global_bin >= 0) & (global_true_bin >= 0)

    # Fill values in vectors
    # inprop.reco = np.concatenate((inprop.reco, reco_value[valid]))
    # inprop.pdg = np.concatenate((inprop.pdg, pdg_id[valid]))
    inprop.added_weights = np.concatenate((inprop.added_weights, mc_weight[valid]))
    inprop.bin_indices = np.concatenate((inprop.bin_indices, global_bin[valid]))
    inprop.trueLE = np.concatenate((inprop.trueLE, true_value[valid]))
    inprop.model_rule = np.concatenate((inprop.model_rule, model_rule[valid]))
    inprop.true_bin_indices = np.concatenate((inprop.true_bin_indices, global_true_bin[valid]))

    # TODO add other, mcStatErr

    # Fill the histogram
    # 
    # IMPORTANT GOTCHA:
    # The folloing code:
    # inprop.hist[global_true_bin[valid], global_bin[valid]] += mc_weight[valid]
    # does not work because in the case of redundant indices, numpy only adds the first one
    np.add.at(inprop.hist, (global_true_bin[valid], global_bin[valid]), mc_weight[valid])

    if not run_syst: return

    # Systematics
    for i,s in enumerate(syst_vector):
        additional_weight = syst_additional_weight[i]
        evw = evws.systematic(s.GetSysName())

        if s.mode == "spline":
            if s.binning == -2:
                spline_bin = global_true_bin
            elif s.binning == -1: 
                spline_bin = global_bin
            else:
                pass # TODO -- implement other binning

            s.FillCV(spline_bin[valid], mc_weight[valid])
            for i_univ, shift in enumerate(s.knobval):
                s.FillUniverse(i_univ, spline_bin[valid], (mc_weight*additional_weight*evw.shift(shift))[valid])
        else:
            s.FillCV(global_bin[valid], mc_weight[valid])
            for i_univ in range(s.GetNUniverse()):
                s.FillUniverse(i_univ, global_bin[valid], (mc_weight*additional_weight*evw.universe(i_univ))[valid])
    # Done!
    return

def process_events(df, fid, syst_structs, evw, prop, c):
    sys_weight_formula = []
    for s in syst_structs:
        if s.HasWeightFormula():
            sys_weight_formula.append(profit.DataFrameFormula("weightMapsFormulas_%i_%s" % (fid, s.GetSysName()), s.GetWeightFormula(), df))

    branches = c.m_branch_variables[fid]
    subchannel_index = []
    for ib, b in enumerate(branches):
        subchannel_name = c.m_branch_variables[fid][ib].associated_hist
        subchannel_index.append(c.GetSubchannelIndex(subchannel_name))

    # Calculate the sys weights
    sys_weight_values = []
    for s in sys_weight_formula:
        sys_weight_values.append(s.EvalInstance())

    # calculate stuff on each event
    for ib, b in enumerate(branches):
        process_branch(c, b, evw, c.m_mcgen_pot[fid], subchannel_index[ib], syst_structs, sys_weight_values, prop)

    # Done!
    return



