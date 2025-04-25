import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import splprep, splev
from rich import print
from multiprocessing import Pool

import data


def dissect_trace(trace, drop_start_index_top, drop_start_index_bot):
    topPart = trace.iloc[:drop_start_index_top]
    midPart = trace.iloc[drop_start_index_top:drop_start_index_bot]
    botPart = trace.iloc[drop_start_index_bot:]
    return topPart, midPart, botPart

def dissection_wrapper(ptrace):
    drop_start_index_top = ptrace[ptrace['z'] > data.ZMAX - data.HEIGHT_TOP_PART - data.HEIGHT_ONE_REP].iloc[0].name
    drop_start_index_bot = ptrace[ptrace['z'] > data.ZMAX - data.HEIGHT_ONE_REP ].iloc[0].name 

    topPart, midPart, botPart = dissect_trace(ptrace, drop_start_index_top, drop_start_index_bot)
    return topPart, midPart, botPart

def interpoolate_bet_parts(topPiece, botPiece, num_interpolation_pts=300, num_data_pts_to_use=30, top_fraction=1/2, bot_fraction=1/2):
    
    # * Divide the top and bottom pieces into head and tail
    topPieceData = topPiece.tail(int(top_fraction*num_data_pts_to_use))
    botPieceData = botPiece.head(int(bot_fraction*num_data_pts_to_use))
    
    dfToInterp = pd.concat([topPieceData, botPieceData])
    x, y, z = dfToInterp['x'].values, dfToInterp['y'].values, dfToInterp['z'].values
    

    # * Perform the spline interpolation
    tck, u = splprep([x, y, z], s=0, k=1)
    new_points = splev(np.linspace(0, 1, num_interpolation_pts), tck)

    interpDF = pd.DataFrame({
        'x': new_points[0], 'y': new_points[1], 'z': new_points[2]
    })
    
    topPieceTrimmed = topPiece.head(len(topPiece)-int(top_fraction*num_data_pts_to_use))
    botPieceTrimmed = botPiece.tail(len(botPiece)-int(bot_fraction*num_data_pts_to_use))

    return interpDF, topPieceTrimmed, botPieceTrimmed

def process_particle_trace(trace_path):
    pnum = int(trace_path.stem.split('_')[-1])
    print(f"Stitching {trace_path} for particle {pnum:02d}")
    ptrace = pd.read_csv(trace_path)

    # * 1st Dissection
    topPart, midPart, botPart = dissection_wrapper(ptrace)

    # * Interpolate the data connecting the pieces to have a finer dissection
    interpTop, topPieceTrimmed, _ = interpoolate_bet_parts(topPart, midPart, num_interpolation_pts=100, num_data_pts_to_use=30, top_fraction=1/3, bot_fraction=2/3)
    interpBot, _, botPieceTrimmed = interpoolate_bet_parts(midPart, botPart, num_interpolation_pts=100, num_data_pts_to_use=30, top_fraction=2/3, bot_fraction=1/3)
    midPartTrimmed = midPart.iloc[int(30*1/2):int(-30*1/2)] # Trim the middle part

    # Reset index
    for df in [topPieceTrimmed, interpTop, midPartTrimmed, interpBot, botPieceTrimmed]:
        df.reset_index(drop=True, inplace=True)

    wholeTrace = pd.concat([topPieceTrimmed, interpTop, midPartTrimmed, interpBot, botPieceTrimmed]).reset_index()

    # * 2nd Dissection (with finer data)
    newTopPart, newMidPart, newBotPart = dissection_wrapper(wholeTrace)

    # * Create copies of the middle piece
    NUM_REPS_DIFF = data.NUM_REPS_INNER_COIL - 2 # 2 loops are already in the top and bottom

    midPartCopies = {i+1: newMidPart.copy() for i in range(NUM_REPS_DIFF)}
    # Shift the middle pieces in y  
    for i in range(NUM_REPS_DIFF):
        midPartCopies[i+1]['z'] += (i)*data.HEIGHT_ONE_REP
    # Shift the bot piece
    newBotPart['z'] += (NUM_REPS_DIFF - 1)*data.HEIGHT_ONE_REP

    # Have a full stack dictionary
    fullStack = {0: newTopPart}
    fullStack.update(midPartCopies)
    fullStack[NUM_REPS_DIFF + 1] = newBotPart

    # * Interpolate the data connecting the pieces
    interpMidPieces = {}
    for i in range(NUM_REPS_DIFF+1):
        
        connectorPiece, upPiece, downPiece = interpoolate_bet_parts(fullStack[i].iloc[:-5], fullStack[i+1].iloc[5:], num_interpolation_pts=50, num_data_pts_to_use=10, top_fraction=1/2, bot_fraction=1/2)        
        
        connectorPieceUP = connectorPiece[connectorPiece['z'] <  data.HEIGHT_TOP_PART + i*data.HEIGHT_ONE_REP]
        connectorPieceDOWN = connectorPiece[connectorPiece['z'] > data.HEIGHT_TOP_PART + i*data.HEIGHT_ONE_REP]
        
        topPiece = pd.concat([upPiece, connectorPieceUP]).reset_index(drop=True)
        bottomPiece = pd.concat([connectorPieceDOWN, downPiece]).reset_index(drop=True)

        fullStack[i+1] = bottomPiece
        interpMidPieces[i] = topPiece

    finalTrace = pd.concat([ piece for piece in interpMidPieces.values()]).reset_index(drop=True)
    finalTrace = pd.concat([finalTrace, fullStack[NUM_REPS_DIFF + 1]]).reset_index(drop=True)
    finalTrace.drop(columns=['index'], inplace=True)
    
    # * Remove duplicates
    # ! These are possibly due to the interpolation
    finalTrace = finalTrace.drop_duplicates(subset=['x', 'y', 'z'], keep='first')
    
    
    finalTrace.to_csv(data.DIR_PATH / 'inner_coil_data_stitched' / f'inner_bcoil_particle_{pnum:02d}_stitched.csv', index=False)
    
def main():
    # using pathlib's glob
    trace_paths = list(data.DIR_PATH.glob('coil_data_trimmed/particle_*.csv'))
    
    with Pool(4) as pool:
        list(tqdm(pool.imap(process_particle_trace, trace_paths), total=len(trace_paths), desc="Processing traces"))
        
if __name__ == '__main__':
    
    pd.options.mode.chained_assignment = None  # Suppress pandas warnings
    main()