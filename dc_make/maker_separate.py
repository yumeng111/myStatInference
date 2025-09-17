import itertools
import math
import os
import re
import yaml

#yumeng
from CombineHarvester.CombineTools.ch import CombineHarvester

from StatInference.common.tools import listToVector, rebinAndFill, importROOT, resolveNegativeBins, getRelevantBins
from .process import Process
from .uncertainty import Uncertainty, UncertaintyType, UncertaintyScale, MultiValueLnNUncertainty
from .model import Model
from .binner import Binner
ROOT = importROOT()

#(1)yumeng:DatacardMaker initialization (what’s loaded from config)
class DatacardMaker:
  def __init__(self, cfg_file, input_path, hist_bins=None, param_values=None):
    #yumeng: central CH object
    self.cb = CombineHarvester()

    #yumeng: Loads YAML into cfg, extract objects
    self.input_path = input_path
    with open(cfg_file, 'r') as f:
      cfg = yaml.safe_load(f)

    self.analysis = cfg["analysis"]
    self.eras = cfg["eras"]
    self.channels = cfg["channels"]
    self.categories = cfg["categories"]
    self.signalFractionForRelevantBins = cfg['signalFractionForRelevantBins']

    # yumeng: name each bin as "<era>_<analysis>_<channel>_<category>", caches them in self.bins
    self.bins = []
    for era, channel, cat in self.ECC():
      bin = self.getBin(era, channel, cat, return_index=False)
      self.bins.append(bin)

    #yumeng:Instantiate model, encapsulate the model parameters(eg parameters, param_dependent_bkg...)
    self.model = Model.fromConfig(cfg["model"])
    
    # yumeng: Build process list (data, bkgs, signals)
    self.param_bins = {}
    self.processes = {}
    data_process = None
    has_signal = False
    for process in cfg["processes"]:
      #suspicious why mass?: for signal dictionary: scan diff param values from cmd line without modify yaml
      if (type(process) != str) and process.get('is_signal', False):
        if param_values is not None:
          print(f"Overwriting signal masses to {param_values}")
          process['param_values'] = param_values
      
      
      #Yumeng: Expand (+ possibly split into subprocesses) and register processes (for top level process and keep track of subprocesses)
      new_processes = Process.fromConfig(process, self.model)
      for process in new_processes:
        if process.name in self.processes:
          raise RuntimeError(f"Process name {process.name} already exists")
        print(f"Adding {process}")
        self.processes[process.name] = process
        #yumeng: only one data process is allowed
        if process.is_data:
          if data_process is not None:
            raise RuntimeError("Multiple data processes defined")
          data_process = process
        #yumeng: suspicious
        if process.is_signal:
          has_signal = True
          param_bin = self.model.paramStr(process.params)
          # yumeng: keep a unique label for each parameter point (param_str)
          # if param_bin in self.param_bins:
          #   raise RuntimeError(f"Signal process with parameters {param_bin} already exists")
          # self.param_bins[param_bin] = process.params
          # change:Allow multiple signals to share same parameter point. will have one CH/param bin per (era, analysis, channel, category), 
          #and separate processes (bin) living inside the same CH/param bin.
          #dict.setdefault(key, default) python built-in function: If key not in dict, inserts key with given default value.i
          #and if yes, returns existing value and does not overwrite.
          self.param_bins.setdefault(param_bin, process.params)
    if data_process is None:
      raise RuntimeError("No data process defined")
    if not has_signal:
      raise RuntimeError("No signal process defined")

    # yumeng: Uncertainties
    self.uncertainties = {}
    for unc_entry in cfg["uncertainties"]:
      unc = Uncertainty.fromConfig(unc_entry)
      if unc.name in self.uncertainties:
        raise RuntimeError(f"Uncertainty {unc.name} already exists")
      self.uncertainties[unc.name] = unc

    #yumeng: 
    # Below autolnNThr, becomes lnN
    # delta (difference between up and down) Below asymlnNThr, treated as symmetric
    # Above ignorelnNThr, included
    self.autolnNThr = cfg.get("autolnNThr", 0.05)
    self.asymlnNThr = cfg.get("asymlnNThr", 0.001)
    self.ignorelnNThr = cfg.get("ignorelnNThr", 0.001)

    self.autoMCStats = cfg.get("autoMCStats", { 'apply': False })


    hist_bins = hist_bins or cfg.get("hist_bins", None)
    # yumeng: Binning helper
    self.hist_binner = Binner(hist_bins)

    # yumeng: caches
    self.input_files = {}
    self.shapes = {}


  def getBin(self, era, channel, category, return_name=True, return_index=True):
    name = f'{era}_{self.analysis}_{channel}_{category}'
    if not return_name and not return_index:
      raise RuntimeError("Invalid argument combination")
    if not return_index:
      return name
    index = self.bins.index(name)
    if not return_name:
      return index
    return (index, name)

  def cbCopy(self, param_str, process, era, channel, category):
    bin_idx, bin_name = self.getBin(era, channel, category)
    return self.cb.cp().mass([param_str]).process([process]).bin([bin_name])

  def ECC(self):
    return itertools.product(self.eras, self.channels, self.categories)

  def PPECC(self):
    param_bins = list(self.param_bins.keys())
    if not self.model.param_dependent_bkg:
      param_bins.append("*")
    return itertools.product(self.processes.keys(), param_bins, self.eras, self.channels, self.categories)

  #(2)yumeng:Get the input file (Each file is opened once and cached in self.input_files) for a given era and model parameters
  def getInputFile(self, era, model_params):
    #suspicious
    #yumeng: builds a file name using the configured pattern and the parameter values (e.g., encodes kl,k2v in the file name)
    file_name = self.model.getInputFileName(era, model_params)
    if file_name not in self.input_files:
      full_file_name = os.path.join(self.input_path, file_name)
      file = ROOT.TFile.Open(full_file_name, "READ")
      if file == None:
        raise RuntimeError(f"Cannot open file {full_file_name}")
      self.input_files[file_name] = file
    return file_name, self.input_files[file_name]


  def getMultiValueLnUnc(self,unc,unc_name, process, era, channel, category, model_params):#, unc_name=None, unc_scale=None)
    file_name, file = self.getInputFile(era, model_params)
    hist_name = f"{channel}/{category}/{process.hist_name}"
    #yumeng: handles MultiValueLnNUncertainty where a single “process” in datacard is sum of subprocess histograms with different lnN numbers.
    #Flow:
    #If uncertainty object has a direct value for full process, return it.
    #Else, if process has subprocesses, it:
    ##Opens each subprocess histogram to get its yield.
    ##Looks up that subprocess’s lnN value (could be dict with Up/Down).
    ##Returns a yield-weighted average of Up and Down across subprocesses.
    #This express single lnN lines for a merged process that is made of heterogeneous components with different lnN magnitudes.
    if unc.getUncertaintyForProcess(process.name) != None:
      return unc.getUncertaintyForProcess(process.name)
    elif process.subprocesses:
      unc_value_tot_down = 0.
      unc_value_tot_up = 0.
      yield_value_tot = 0.
      for subp in process.subprocesses:
        hist_name = f"{channel}/{category}/{subp}"
        subhist = file.Get(hist_name)
        #newhist = self.hist_binner.applyBinning(era, channel, category, model_params, subhist)
        if subhist == None:
          raise RuntimeError(f"Cannot find histogram {hist_name} in {file.GetName()}")
        axis = subhist.GetXaxis()
        yield_subproc = subhist.Integral(1,axis.GetNbins() + 1)
        unc_value = unc.getUncertaintyForProcess(subp)
        if unc_value != None:
          if yield_subproc == 0 : continue
          # print(unc_value)
          if isinstance(unc_value, dict):
            unc_value_tot_up += unc_value[UncertaintyScale.Up]*yield_subproc
            unc_value_tot_down += unc_value[UncertaintyScale.Down]*yield_subproc
          else:
            unc_value_tot_up += unc_value*yield_subproc
            unc_value_tot_down -= unc_value*yield_subproc
          yield_value_tot+=yield_subproc
      if unc_value_tot_up != 0. and unc_value_tot_down !=0 :
        return {UncertaintyScale.Down: unc_value_tot_down/yield_value_tot, UncertaintyScale.Up: unc_value_tot_up/yield_value_tot}
      return None
    return None


  # (3)yumeng: core, how histograms are extracted and got the shape for a given process, era, channel, category, and model parameters
  def getShape(self, process, era, channel, category, model_params, unc_name=None, unc_scale=None):
    file_name, file = self.getInputFile(era, model_params)
    #change: drop signal_processes_histograms
    #yumeng: (a) Pick & cache by a stable key
    # change: Add parameter tag to avoid cache key collisions across parameter points, eg kl_2_k2v_1
    param_tag = None if model_params is None else self.model.paramStr(model_params)
    key = (file_name, process.name, era, channel, category, unc_name, unc_scale, param_tag)
    if key not in self.shapes:
      # yumeng: (b) Asimov data is handled specially = sum of background shapes, because?there is no file.Get for a "data" histogram
      if process.is_data and (unc_name is not None or unc_scale is not None):
        raise RuntimeError("Cannot apply uncertainty to the data process")
      if process.is_asimov_data:
        hist = None
        for bkg_proc in self.processes.values():
          if bkg_proc.is_background:
            bkg_hist = self.getShape(bkg_proc, era, channel, category, model_params)
            if hist is None:
              hist = bkg_hist.Clone()
            else:
              hist.Add(bkg_hist)
        if hist is None:
          raise RuntimeError("Cannot create asimov data histogram")
      else:
        #yumeng: (c) Nominal or shifted (shape) histogram name(s)
        # base: "<channel>/<category>/<process.hist_name or subprocess>"
        # if it's a shape uncertainty: append "_{unc_name}{Up/Down}"
        
        # which histogram name to look up in the ROOT file
        hist_name = f"{channel}/{category}/{process.hist_name}"
        hists = []
        if process.subprocesses:
          # yumeng:Collect and sum sub-hists
          for subp in process.subprocesses:
            hist_name = f"{channel}/{category}/{subp}"
            if unc_name and unc_scale:
              hist_name += f"_{unc_name}{unc_scale}"
            subhist = file.Get(hist_name)
            if subhist == None:
              raise RuntimeError(f"Cannot find histogram {hist_name} in {file.GetName()}")
            # yumeng: rebin
            hists.append(self.hist_binner.applyBinning(era, channel, category, model_params, subhist))
        else:
          # # change: Append uncertainty suffix for non-subprocess shapes
          # if unc_name and unc_scale:
          #   hist_name += f"_{unc_name}{unc_scale}"
          hist = file.Get(hist_name)
          if hist == None:
            raise RuntimeError(f"Cannot find histogram {hist_name} in {file.GetName()}")
          hists.append(self.hist_binner.applyBinning(era, channel, category, model_params, hist))
        if len(hists) == 0:
          raise RuntimeError(f"hist list is empty for file {file.GetName()}")
        
        # yumeng: (d) Sum (if many), then finalize name/title
        hist = hists[0]
        if len(hists)>1:
          for histy in hists[1:]:
            hist.Add(histy)
        hist.SetName(process.name)
        hist.SetTitle(process.name)

        # yumeng: (e) Detach from file directory; apply per-process scale
        hist.SetDirectory(0)
        if process.scale != 1:
          hist.Scale(process.scale)
        #yumeng: Signal, just append to a local list (used to decide "relevant bins").
        if process.is_signal:
            # change: Always store signal histogram under nominal key for consistent relevant bins calculation
            nominal_signal_key = (file_name, "signals", era, channel, category, None, None)
            self.shapes.setdefault(nominal_signal_key, []).append(hist)
        #yumeng: 
        #explain: it's doing: When processing a signal process: stores signal histogram in special cache location: 
        #Cache key: (file_name, "signals", era, channel, category, None, None) - note "signals" string and None, None for uncertainty parameters
        #Purpose: This creates collection of all signal histograms for a given (file, era, channel, category) combination
        #Why it's needed: Later, when processing background processes, code needs to know which bins are "relevant" (where signal is significant)
        #to decide how to handle negative bins:
        #summary: 1) Signal processing: Store all signal histograms in special cache
        #2) Background processing: Retrieve those signal histograms to calculate which bins are "relevant" (where signal fraction > threshold)
        #3) Negative bin remediation: Only apply strict negative bin handling to "relevant" bins where signal matters

        #Background, compute a relevant-bins mask (bins where signal fraction exceeds a threshold), 
        #then run negative-bin remediation (clip/smooth/merge per resolveNegativeBins policy).
        #If remediation fails, prints edges/values/errors and throws to force a fix upstream.
        #idea: only care about negative bins where signal actually matters (by signalFractionForRelevantBins threshold). reduces false alarms in empty tails.
        else:
          # change: Always use nominal signals for relevant bins calculation, regardless of uncertainty
          nominal_signal_key = (file_name, "signals", era, channel, category, None, None)
          signal_processes_histograms = self.shapes.get(nominal_signal_key, [])
          relevant_bins = getRelevantBins(era, channel, category,signal_processes_histograms,self.signalFractionForRelevantBins,unc_name, unc_scale, model_params)
          solution = resolveNegativeBins(hist,relevant_bins=relevant_bins, allow_zero_integral=process.allow_zero_integral, allow_negative_bins_within_error=process.allow_negative_bins_within_error, max_n_sigma_for_negative_bins=process.max_n_sigma_for_negative_bins, allow_negative_integral=process.allow_negative_integral)

          if not solution.accepted:
            axis = hist.GetXaxis()
            bins_edges = [ str(axis.GetBinLowEdge(n)) for n in range(1, axis.GetNbins() + 2)]
            bin_values = [ str(hist.GetBinContent(n)) for n in range(1, axis.GetNbins() + 1)]
            bin_errors = [ str(hist.GetBinError(n)) for n in range(1, axis.GetNbins() + 1)]
            print(f'bins_edges: [ {", ".join(bins_edges)} ]')
            print(f'bin_values: [ {", ".join(bin_values)} ]')
            print(f'bin_errors: [ {", ".join(bin_errors)} ]')
            raise RuntimeError(f"Negative bins found in histogram {hist_name}")
      self.shapes[key] = hist
    return self.shapes[key]




  #(4)yumeng:addProcess & attaching shapes to CombineHarvester
  def addProcess(self, proc, era, channel, category):
    bin_idx, bin_name = self.getBin(era, channel, category)
    process = self.processes[proc]
    #suspicious, AddObservations or AddProcesses declare bins and process roster to CH.
    # change: Modified add function to accept process_name parameter for unique signal names
    def add(model_params, param_str, process_name):
      if process.is_data:
        self.cb.AddObservations([param_str], [self.analysis], [era], [channel], [(bin_idx, bin_name)])
      else:
        #change:
        self.cb.AddProcesses([param_str], [self.analysis], [era], [channel], [process_name], [(bin_idx, bin_name)], process.is_signal)

      # yumeng: Fetch nominal shape (calls routine getShape)
      shape = self.getShape(process, era, channel, category, model_params)
      # yumeng: Stash shape into CH for this (process, bin, param))
      shape_set = False
      def setShape(p):
        nonlocal shape_set
        print(f"Setting shape for {p}")
        if shape_set:
          raise RuntimeError("Shape already set")
        p.set_shape(shape, True) # yumeng: << attaches TH1 as nominal template for that (param_str, process, bin).
        shape_set = True
      #change:
      cb_copy = self.cbCopy(param_str, process_name, era, channel, category)
      if process.is_data:
        cb_copy.ForEachObs(setShape)
      else:
        cb_copy.ForEachProc(setShape)

    # yumeng: Suspicious: For signals: iterate over each parameter point (mass = param_str)
    # change: Modified signal processing to create unique process names and use "*" mass
    if process.is_signal:
      model_params = process.params
      param_str = self.model.paramStr(model_params)
      # change: Use the process name as-is since it already includes parameters from Process.fromConfig, eg ggHH_kl_1_kt_1_hbbhtt
      unique_proc_name = proc  # Process name already includes parameters
      add(model_params, "*", unique_proc_name)  # Use "*" mass for all processes
    elif self.model.param_dependent_bkg:
      for signal_proc in self.processes.values():
        if signal_proc.is_signal:
          model_params = signal_proc.params
          param_str = self.model.paramStr(model_params) # yumeng: # e.g. "kl_1__k2v_0"
          #change:
          add(model_params, "*", proc)  # Use "*" mass for all processes
    else:
      # yumeng: Backgrounds (and data) are stored under '*' unless bkg depends on params
      #change:
      add(None, "*", proc)

  #(5)yumeng:addUncertainties (including shape systematics)
  def addUncertainty(self, unc_name):
    unc = self.uncertainties[unc_name]
    isMVLnUnc = isinstance(unc, MultiValueLnNUncertainty)
    for proc, param_str, era, channel, category in self.PPECC():
      process = self.processes[proc]
      if process.is_data: continue
      # yumeng: Get model parameters for current parameter point (dict of kl,k2v for signals; None for '*')
      model_params = self.param_bins.get(param_str, None)
      if isMVLnUnc:
        unc_value = self.getMultiValueLnUnc(unc,unc_name,process, era, channel, category, model_params)#, unc_name=None, unc_scale=None

      uncApplies = unc_value != None if isMVLnUnc else unc.appliesTo(process, era, channel, category)
      if not uncApplies: continue
      if not process.hasCompatibleModelParams(model_params, self.model.param_dependent_bkg): continue

      # change: Determine the actual process name used in CombineHarvester. * for mass
      if process.is_signal:
        actual_proc_name = proc  # Process name already includes parameters from Process.fromConfig
        actual_param_str = "*"
      else:
        actual_proc_name = proc
        actual_param_str = "*"

      nominal_shape = None
      shapes = {}
      if unc.needShapes:
        model_params = self.param_bins.get(param_str, None)
        # yumeng: Prepare nominal, up, down shapes and attach them
        # For lnN style, just registers numbers.
        #For shape style, calls getShape(..., unc_name, Up/Down) to fetch shifted TH1s and sets them on CH systematic with set_shapes(up, down, nominal).
        nominal_shape = self.getShape(self.processes[proc], era, channel, category, model_params)
        for unc_scale in [ UncertaintyScale.Up, UncertaintyScale.Down ]:
          shapes[unc_scale] = self.getShape(self.processes[proc], era, channel, category, model_params,
                                            unc_name, unc_scale.name)
      unc_to_apply = unc.resolveType(nominal_shape, shapes, self.autolnNThr, self.asymlnNThr)
      can_ignore = unc_to_apply.canIgnore(unc_value, self.ignorelnNThr) if isMVLnUnc else unc_to_apply.canIgnore(self.ignorelnNThr)
      if can_ignore:
        print(f"Ignoring uncertainty {unc_name} for {proc} in {era} {channel} {category}")
        continue
      systMap = unc_to_apply.valueToMap(unc_value) if isMVLnUnc else unc_to_apply.valueToMap()
      # yumeng: Build a SystMap (lnN/shape/asym, possibly multi-value per subprocess)
      # change: Use actual process name and mass parameter
      cb_copy = self.cbCopy(actual_param_str, actual_proc_name, era, channel, category)
      cb_copy.AddSyst(self.cb, unc_name, unc_to_apply.type.name, systMap)
      if unc_to_apply.type == UncertaintyType.shape:
        shape_set = False
        def setShape(syst):
          nonlocal shape_set
          print(f"Setting unc shape for {syst}")
          if shape_set:
            raise RuntimeError("Shape already set")
          # yumeng: syst.set_shapes(up, down, nominal)
          syst.set_shapes(shapes[UncertaintyScale.Up], shapes[UncertaintyScale.Down], nominal_shape)
          shape_set = True
        # change: Use actual process name and mass parameter
        cb_copy = self.cbCopy(actual_param_str, actual_proc_name, era, channel, category).syst_name([unc_name])
        cb_copy.ForEachSyst(setShape)

  #(6)yumeng:writeDatacards (writes datacards and shape files)
  def writeDatacards(self, output):
    os.makedirs(output, exist_ok=True)
    
    # change: Simplified process collection - get all process names from CombineHarvester object
    # Get all process names from the CombineHarvester object (these now include unique signal names)
    all_process_names = set()
    for proc in self.cb.cp().process_set():
      all_process_names.add(proc)
    
    # change: Create separate datacards for each channel-category combination
    def slug(s: str) -> str:
      # lower, turn slashes into underscores, and collapse any weird chars to "_"
      s = s.lower().replace('/', '_')
      return re.sub(r'[^a-z0-9_.-]+', '_', s)
    
    for subera in self.eras:
      for subchannel in self.channels:
        for subcat in self.categories:
          channel_lower = subchannel.lower()               # eTau -> etau
          cat_slug = slug(subcat)                          # OS_Iso/res2b_cat3 -> os_iso_res2b_cat3
          dc_name = f"{channel_lower}_{cat_slug}"          # etau_os_iso_res2b_cat3

          # Output dir per (era, combo)
          tmp_output = os.path.join(output, subera, dc_name)
          os.makedirs(tmp_output, exist_ok=True)

          # Files
          tmp_dc_file = os.path.join(tmp_output, f"datacard_{dc_name}.txt")
          tmp_shape_file = os.path.join(tmp_output, f"shapes_{dc_name}.root")

          # Select exactly this bin (real CH name with slash is fine)
          bin_name = self.getBin(subera, subchannel, subcat, return_index=False)
          self.cb.cp().era([subera]).channel([subchannel]).bin([bin_name]).mass(['*']).WriteDatacard(
              tmp_dc_file, tmp_shape_file
          )

    # change: Write main combined datacard with all signals and backgrounds
    # Write the main combined datacard (all eras/channels/categories)
    dc_file = os.path.join(output, "datacard_combined_signals.txt")
    # change: Create main combined shape file
    main_shape_file = os.path.join(output, "combined_signals_all.root")
    #change:
    self.cb.cp().mass(['*']).WriteDatacard(dc_file, main_shape_file)



  def createDatacards(self, output, verbose=1):
    try:
      for era, channel, category in self.ECC():
        for process_name in self.processes.keys():
          self.addProcess(process_name, era, channel, category)
      for unc_name in self.uncertainties.keys():
        print(f"adding uncertainty: {unc_name}")
        self.addUncertainty(unc_name)
      if self.autoMCStats["apply"]:
        self.cb.SetAutoMCStats(self.cb, self.autoMCStats["threshold"], self.autoMCStats["apply_to_signal"],
                               self.autoMCStats["mode"])
      if verbose > 0:
        self.cb.PrintAll()
      self.writeDatacards(output)
    finally:
      for file in self.input_files.values():
        file.Close()
