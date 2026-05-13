# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 11:22:07 2026

@author: timon_kalchmayr
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import warnings

def parse_result_csv(file_path: str | Path) -> pd.DataFrame:
    """
    Parse .csv file created by Unicorn export.
    
    Supports UV 1, Cond, Conc B, Injection, pH, Run Log column parsing.

    Parameters
    ----------
    file_path : str | Path
        Path to the input file.

    Returns
    -------
    df_return : pandas.DataFrame
        DataFrame containing x and y (Signal) coordinates of the detected columns.

    """
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"File {file_path} does not exist.")
        
    df = pd.read_csv(file_path, skiprows=1)

    uv_1, uv_1_signal = pd.Series(), pd.Series()
    cond, cond_signal = pd.Series(), pd.Series()
    conc, conc_signal = pd.Series(), pd.Series()
    inj = pd.Series()
    ph, ph_signal = pd.Series(), pd.Series()

    for n, col in enumerate(df.columns):
        if col.startswith("UV 1"):      # .startswith() -> make wavelength flexible
            uv_1 = df.iloc[1:, n].astype(float)
            uv_1.name = f"UV{col[-3:]}"
            uv_1_signal = df.iloc[1:, n+1].astype(float)
            uv_1_signal.name = f"{uv_1.name}Signal"
        
        if col == "Cond":
            cond = df.iloc[1:, n].astype(float)
            cond.name = "Conductivity"
            cond_signal = df.iloc[1:, n+1].astype(float)
            cond_signal.name = f"{cond.name}Signal"
            
        if col == "Conc B":
            conc = df.iloc[1:, n].astype(float)
            conc.name = "ConcB"
            conc_signal = df.iloc[1:, n+1].astype(float)
            conc_signal.name = f"{conc.name}Signal"
            
        if col == "Injection":
            inj = df.iloc[1:, n].astype(float)
            inj.name = "InjectionMarker"
            
        if col == "pH":
            ph = df.iloc[1:, n].astype(float)
            ph.name = "pH"
            ph_signal = df.iloc[1:, n+1].astype(float)
            ph_signal.name = f"{ph.name}Signal"
            
        if col == "Run Log":
            log = df.iloc[1:, n].astype(float)
            log.name = "RunLog"
            log_signal = df.iloc[1:, n+1].astype(str)
            log_signal.name = f"{log.name}Signal"
            
    df_return = pd.DataFrame()

    if not (uv_1.empty or uv_1_signal.empty):
        df_return = pd.concat([df_return,
                               uv_1.to_frame(),
                               uv_1_signal.to_frame()],
                               axis=1)
    if not (cond.empty or cond_signal.empty):
        df_return = pd.concat([df_return,
                               cond.to_frame(),
                               cond_signal.to_frame()],
                               axis=1)
        
    if not (conc.empty or conc_signal.empty):
        df_return = pd.concat([df_return,
                               conc.to_frame(),
                               conc_signal.to_frame()],
                               axis=1)
        
    if not (ph.empty or ph_signal.empty):
        df_return = pd.concat([df_return,
                               ph.to_frame(),
                               ph_signal.to_frame()],
                               axis=1)
        
    if not inj.empty:
        df_return = pd.concat([df_return,
                               inj.to_frame()],
                               axis=1)
        
    if not (log.empty or log_signal.empty):
        df_return = pd.concat([df_return,
                               log.to_frame(),
                               log_signal.to_frame()],
                               axis=1)
        
    return df_return


def plot_aekta_chromatogram(df: pd.DataFrame,
                            x_label: str,
                            x_range: tuple[float] = (None, None),
                            plot_uv: bool = True, 
                            plot_cond: bool = True,
                            plot_conc_b: bool = True, 
                            plot_ph: bool = True,
                            ax: matplotlib.axes.Axes = None) -> tuple:
    """
    Plot AEKTA Chromatogram.

    Parameters
    ----------
    df : pd.DataFrame
        Chromatogram data, expected to be created by 
        ``AektaChromatogram.parse_result_csv()``.
    x_label : str
        Label displayed at the x-axis.
    x_range : tuple[float], optional
        Range of data to plot (based on the unit on the x-Axis). None means
        all data is plotted in this direction. The default is (None, None).
    plot_uv : bool, optional
        Set True to plot UV signal. The default is True.
    plot_cond : bool, optional
        Set True to plot conductivity signal. The default is True.
    plot_conc_b : bool, optional
        Set True to plot gradient. The default is True.
    plot_ph : bool, optional
        Set True to plot pH. The default is True.
    ax : matplotlib.axes.Axes, optional
        To plot the chromatogram in a figure containing multiple subplots, pass
        the axis where the plot should be placed. The default is None.

    Raises
    ------
    ValueError

    Returns
    -------
    tuple
        Tuple of matplotlib.figure.Figure, matplotlib.axes.Axes of the plot.

    """
    if not isinstance(x_label, str):
        raise ValueError("x-axis Label must be of type str")
    
    if ax is None:
        fig, ax1 = plt.subplots(1, 1, figsize=(10,5))
    else:
        fig = ax.figure
        ax1 = ax
        
    ax2 = ax1.twinx()
    
    ax1_ylabel = ""
    ax2_ylabel = ""
    
    if plot_uv:
        uv_col = df.columns[df.columns.str.contains("UV") & ~df.columns.str.contains("Signal")]
        if len(uv_col) > 1:
            raise ValueError("Function currently supports only one UV signal")
        start = df[df[uv_col] >= x_range[0]].first_valid_index()
        end = df[df[uv_col] <= x_range[1]][::-1].first_valid_index()
        uv1, = ax1.plot(df.loc[start:end, uv_col],
                        df.loc[start:end, uv_col+"Signal"],
                        label=f"{uv_col[0]} (mAU)",
                        color="#0072B2")
        ax1_ylabel += f"{uv_col[0]} (mAU)" if len(ax1_ylabel)==0 else f", {uv_col[0]} (mAU)"
    
    if plot_cond:
        cond_col = "Conductivity"
        start = df[df[cond_col] >= x_range[0]].first_valid_index()
        end = df[df[cond_col] <= x_range[1]][::-1].first_valid_index()
        cond1, = ax2.plot(df.loc[start:end, cond_col],
                        df.loc[start:end, cond_col+"Signal"],
                        label=f"{cond_col[0]} (mS/cm)",
                        color="black")
        ax2_ylabel += f"{cond_col} (mS/cm)" if len(ax2_ylabel)==0 else f", {cond_col} (mS/cm)"
        
    if plot_conc_b:
        conc_col = "ConcB"
        start = df[df[conc_col] >= x_range[0]].first_valid_index()
        end = df[df[conc_col] <= x_range[1]][::-1].first_valid_index()
        conc1, = ax2.plot(df.loc[start:end, conc_col],
                        df.loc[start:end, conc_col+"Signal"],
                        label="Gradient (%B)",
                        color="#E69F00")
        ax2_ylabel += "Gradient (%B)" if len(ax2_ylabel)==0 else ", Gradient (%B)"
    
    if plot_ph:
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("axes", 1.1))
        ph_col = "pH"
        start = df[df[ph_col] >= x_range[0]].first_valid_index()
        end = df[df[ph_col] <= x_range[1]][::-1].first_valid_index()
        ph1, = ax3.plot(df.loc[start:end, ph_col],
                        df.loc[start:end, ph_col+"Signal"],
                        label="pH",
                        color="#CC79A7")
        ax3.set_ylabel("pH")
        ax3.set_ylim(2, 10)
        
        
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(ax1_ylabel)
    ax2.set_ylabel(ax2_ylabel)
    
    if plot_ph:
        axes = (ax1, ax2, ax3)
    else:
        axes = (ax1, ax2)
    
    return fig, axes


class AektaChromatogram:
    """
    Class that holds an ÄKTA chromatogram and has methods to parse and modify it.
    """
    def __init__(self, file_path: str | Path) -> None:
        self.data = parse_result_csv(file_path)
        self.performed_modifications = []
                
    
    def correct_volume_from_run_log(self, mark: str) -> None:
        """
        Set x-axes to 0 at certain point of run log.
    
        Parameters
        ----------
        mark : str
            Start of the run log entry where the volume should be set to 0.
    
        Raises
        ------
        ValueError
            If one of the expected columns (InjectionMark, RunLog, RunLogSignal)
            is not within the provided DataFrame   
        """
        if not "InjectionMarker" in self.data:
            raise ValueError("Expected column InjectionMark not in provided DataFrame")
        if not "RunLog" in self.data:
            raise ValueError("Expected column RunLog not in provided DataFrame")
        if not "RunLogSignal" in self.data:
            raise ValueError("Expected column RunLogSignal not in provided DataFrame")
        if "Volume corrected from Run Log" in self.performed_modifications:
            warnings.warn("Volume correction from Run Log was already performed, check if it makes sense to perform it again")
        if "Volume corrected from Injection" in self.performed_modifications:
            warnings.warn("Volume was already corrected from injection, check if it makes sense to additionally correct from Run Log")
        
        inj = self.data["InjectionMarker"].iloc[0]
        log_v_ind = self.data[self.data["RunLogSignal"].str.contains(mark)].index
        if len(log_v_ind) > 1:
            raise ValueError("Provided mark was ambiguous and returned multiple \
                             run log entries")
        
        log_v = self.data["RunLog"].iloc[log_v_ind].iloc[0]    
        corr_v = log_v - inj
        
        cols_to_correct = self.data.columns[~self.data.columns.str.contains("Signal")]
        
        for c in cols_to_correct:
            self.data.loc[:, c] = self.data.loc[:, c] - corr_v
            
        self.performed_modifications.append("Volume corrected from Run Log")
        
        
    def correct_volume_from_injection(self, inj_no: int = 0) -> None:
        """

        Parameters
        ----------
        inj_no : int, optional
            Injection mark number (0 indexing). The default is 0 
            ( uses the first injection mark).
        """
        if not "InjectionMarker" in self.data:
            raise ValueError("Expected column InjectionMark not in provided DataFrame")
        if inj_no > len(self.data["InjectionMarker"]) or inj_no < 0:
            raise ValueError("Provided Injection mark index not available in data.")
        if "Volume corrected from Run Log" in self.performed_modifications:
            warnings.warn("Volume already corrected from Run Log, check if it makes sense to additionally correct from Injection")
        if "Volume corrected from Injection" in self.performed_modifications:
            warnings.warn("Volume correction from Injection was already performed, check if it makes sense to perform it again")
        
        
        corr_v = self.data["InjectionMarker"].iloc[inj_no]
        
        cols_to_correct = self.data.columns[~self.data.columns.str.contains("Signal")]
        
        for c in cols_to_correct:
            self.data.loc[:, c] = self.data.loc[:, c] - corr_v
            
        self.performed_modifications.append("Volume corrected from Injection")
            
    def correct_for_delay_volume(self, columns : list[str], 
                                 v_delay: float) -> None:
        """
        Corrects selected data and signal column for delay volum (in mL).
        Directly works on the DataFrame of the class instance.

        Parameters
        ----------
        cols : list[str]
            Names of the columns to correct in self.data DataFrame.
        v_delay : float
            Delay volume to subtract in mL.

        """
        if not all(col in self.data.columns for col in columns):
            raise ValueError("One or more provided columns not in chromatogram data")
        
        for c in columns:
            self.data.loc[:, c] = self.data.loc[:, c] - v_delay
            
        self.performed_modifications.append(f"Corrected columns {columns} for delay volume {v_delay}.")
        
    def x_axes_to_cv(self, cv: float) -> None:
        """
        Convert unit of x-axis to column volumes.
    
        Parameters
        ----------
        cv : float
            Volume of the column.    
        """
        if "Converted x-axis to CV" in self.performed_modifications:
            warnings.warn("X-axis is already converted.")
            return
        
        cols_to_correct = self.data.columns[~self.data.columns.str.contains("Signal")]
        
        for c in cols_to_correct:
            self.data.loc[:, c] = self.data.loc[:, c] / cv
            
        self.performed_modifications.append("Converted x-axis to CV")
            
    
    def plot(self, x_label, **kwargs) -> tuple:
        """
        Wrap plot_aekta_chromatogram to directly plot data of class instance.
        
        Parameters
        ----------
        x_label : str
            Label of x-axis
        **kwargs
            Arguments passed to ``plot_aekta_chromatogram()``
            
        Returns
        -------
        tuple
            Tuple of matplotlib.figure.Figure, matplotlib.axes.Axes of the plot.
        """
        return plot_aekta_chromatogram(self.data, x_label, **kwargs)