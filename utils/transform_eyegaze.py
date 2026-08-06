import numpy as np
import pandas as pd


def ypr_to_R_RH_ZYX(yaw_deg, pitch_deg, roll_deg): # rotation in z, y, x
    y, p, r = np.deg2rad([yaw_deg, pitch_deg, roll_deg])
    cy, sy = np.cos(y), np.sin(y)
    cp, sp = np.cos(p), np.sin(p)
    cr, sr = np.cos(r), np.sin(r)
    Rz = np.array([[ cy, -sy, 0],[ sy, cy, 0],[0,0,1]])
    Ry = np.array([[ cp, 0, sp],[ 0, 1, 0],[-sp, 0, cp]])
    Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    return Rz @ Ry @ Rx  # cam->world

def ypr_to_R_LH_ZYX(yaw_deg, pitch_deg, roll_deg):
    # LH rotation using Z(-yaw) Y(pitch) X(-roll)
    return ypr_to_R_RH_ZYX(-yaw_deg, pitch_deg, -roll_deg)
    # return ypr_to_R_RH_ZYX(yaw_deg, pitch_deg, roll_deg)

def project_LH_pure(Pw, Cw, yaw_deg, pitch_deg, roll_deg, fx, fy, cx, cy):
    Rlh = ypr_to_R_LH_ZYX(yaw_deg, pitch_deg, roll_deg)  # cam->world (LH)
    Pc = Rlh.T @ (Pw - Cw)  # world->cam (LH)
    # print(Rlh.T)
    X, Y, Z = Pc
    # print('\nray:', Pw - Cw)
    # print(f'Pc: {Pc}\n')
    # if Z <= 0:
    #     return None
    # # flip to RH just for pixel mapping
    # u = fx*(X/Z) + cx
    # v = -fy*((-Y)/Z) + cy  # note the -Y
    if X <= 0:
        return None
    # flip to RH just for pixel mapping
    # u = fx*(Y/X) + cx
    # v = -fy*((-Z)/X) + cy  # note the -Y
    u = fx*(-Y/X) + cx
    v = fy*((Z)/X) + cy  # note the -Y
    # u = fx*(Y/X) + cx
    # v = fy*((Z)/X) + cy  # note the -Y
    # print(u-cx, v-cy)
    return u, v

## hit on neighbor
# Cw = np.array([-16527.04568, 173.973021, 148.171997]) / 100
# # Pw = np.array([-19019.535932, -535.046528, 95.831219]) / 100
# Pw = np.array([-18800.0, -100.0, 0.0]) / 100
# # Pw = np.array([-17106.581466, -364.415357, 2.647764]) / 100
# # Pw = np.array([-18445.707365, -929.083061, 77.167837]) / 100
# # Pw = np.array([-18605.197537, -984.90718, 90.56818]) / 100
# yaw_deg = -150.853806
# pitch_deg = -1.386804
# roll_deg = 3.028832
# yaw_deg *= -1 # as the recorded rotation is not in conventional frame

## hit on pod 
# Cw = np.array([-16528.275664, 171.054235, 147.805756]) / 100
# # Pw = np.array([-18660.0, -83.182331, 122.80146]) / 100
# # Pw = np.array([-18800.0, -100.0, 0.0]) / 100
# yaw_deg = -158.896881
# pitch_deg = 1.189202
# roll_deg = 2.676703
# Cw = Cw * np.array([1, -1, 1])
# Pw = Pw * np.array([1, -1, 1])
# pitch_deg = -pitch_deg

## hit on goal
# Cw = np.array([-16704.950546, 1.634704, 144.808823]) / 100  
# Pw = np.array([-16966.491998, -252.045056, 2.7125]) / 100
# yaw_deg = -151.537933
# pitch_deg = -10.921031
# roll_deg = 6.607353
# yaw_deg *= -1 


## test
# Cw = np.array([0, 0, 0])
# Pw = Cw + np.array([0, 100, 0])
# yaw_deg = -45
# pitch_deg = 0
# roll_deg = 0


# fov_h = 107.21
# fov_v = 107.82
# width = 2880
# height = 1600
# fx = (width/2) / np.tan(np.deg2rad(fov_h)/2)
# fy = (height/2) / np.tan(np.deg2rad(fov_v)/2)
# cx = width / 2
# cy = height / 2
# projected = project_LH_pure(Pw, Cw, yaw_deg, pitch_deg, roll_deg, fx, fy, cx, cy)
# print("projected raw:", projected)

# if projected is not None:
#     width_cap = 1536
#     height_cap = 1068
#     u_cap = width_cap - projected[0] * (width_cap / width)
#     v_cap = height_cap - projected[1] * (height_cap / height)
#     print("projected in capture:", (u_cap, v_cap))


def cal_eyegaze_on_screen(df_ps):
    df_ps = df_ps.copy()
    # eye gaze data
    Cw = df_ps[['Ped_Location_x', 'Ped_Location_y', 'Ped_Location_z']].to_numpy() / 100 # ped loc
    Pw = df_ps[['HitLocation_x', 'HitLocation_y', 'HitLocation_z']].to_numpy() / 100 # gaze hit
    yaw_deg = df_ps[['Ped_Rotation_z']].to_numpy().flatten() * -1
    pitch_deg = df_ps[['Ped_Rotation_y']].to_numpy().flatten()
    roll_deg = df_ps[['Ped_Rotation_x']].to_numpy().flatten()

    # camera instrincs 
    fov_h = 107.21
    fov_v = 107.82
    width = 2880
    height = 1600
    fx = (width/2) / np.tan(np.deg2rad(fov_h)/2)
    fy = (height/2) / np.tan(np.deg2rad(fov_v)/2)
    cx = width / 2
    cy = height / 2

    # calculate using a pinhole camera model
    xs = []
    ys = []
    us = []
    vs = []
    width_cap = 1536
    height_cap = 1068
    for i in range(len(Cw)):
        # print(f"--- point {i} ---")
        projected = project_LH_pure(Pw[i], Cw[i], yaw_deg[i], pitch_deg[i], roll_deg[i], fx, fy, cx, cy)
        # print("projected raw:", projected)
        xs.append(projected[0] if projected is not None else np.nan)
        ys.append(projected[1] if projected is not None else np.nan)

        if projected is not None:
            u_cap = width_cap - projected[0] * (width_cap / width)
            v_cap = height_cap - projected[1] * (height_cap / height)
            # print("projected in capture:", (u_cap, v_cap))
            us.append(u_cap)
            vs.append(v_cap)
        else:
            us.append(np.nan)
            vs.append(np.nan)
    df_ps.loc[:, ['px_x']] = xs
    df_ps.loc[:, ['px_y']] = ys
    df_ps.loc[:, ['px_u']] = us
    df_ps.loc[:, ['px_v']] = vs
    df_ps.loc[:, ['timestamp']] = df_ps['TimeElapsed'] - df_ps['TimeElapsed'].iloc[0]
    df_ps_new = df_ps[['px_u', 'px_v']]

    # get rid of the outliers by clipping
    df_ps_new.loc[:, 'px_u'] = df_ps_new['px_u'].clip(0, width_cap)
    df_ps_new.loc[:, 'px_v'] = df_ps_new['px_v'].clip(0, height_cap)
    return df_ps_new


def test_server():
    pid, sid = 1, 2 # 'P1S2'
    csv_filepath = '../data/dfs_combined.csv'
    df = pd.read_csv(csv_filepath)
    df_ps = df[(df['pid'] == pid)& (df['sid'] == sid)]
    df_ps = cal_eyegaze_on_screen(df_ps)
    return df_ps


def test_local():
    csv_filepath = 'data/vrdata/combined/P1_20241217103615516.csv'
    df_p = pd.read_csv(csv_filepath)
    sid = '3'
    df_ps = df_p[df_p['Scenario'] == sid]
    Cw = df_ps[['Ped_Location_x', 'Ped_Location_y', 'Ped_Location_z']].to_numpy() / 100
    Pw = df_ps[['HitLocation_x', 'HitLocation_y', 'HitLocation_z']].to_numpy() / 100
    yaw_deg = df_ps[['Ped_Rotation_z']].to_numpy().flatten() * -1
    pitch_deg = df_ps[['Ped_Rotation_y']].to_numpy().flatten()
    roll_deg = df_ps[['Ped_Rotation_x']].to_numpy().flatten()

    fov_h = 107.21
    fov_v = 107.82
    width = 2880
    height = 1600
    fx = (width/2) / np.tan(np.deg2rad(fov_h)/2)
    fy = (height/2) / np.tan(np.deg2rad(fov_v)/2)
    cx = width / 2
    cy = height / 2

    xs = []
    ys = []
    us = []
    vs = []
    for i in range(len(Cw)):
        # print(f"--- point {i} ---")
        projected = project_LH_pure(Pw[i], Cw[i], yaw_deg[i], pitch_deg[i], roll_deg[i], fx, fy, cx, cy)
        # print("projected raw:", projected)
        xs.append(projected[0] if projected is not None else np.nan)
        ys.append(projected[1] if projected is not None else np.nan)

        if projected is not None:
            width_cap = 1536
            height_cap = 1068
            u_cap = width_cap - projected[0] * (width_cap / width)
            v_cap = height_cap - projected[1] * (height_cap / height)
            # print("projected in capture:", (u_cap, v_cap))
            us.append(u_cap)
            vs.append(v_cap)
        else:
            us.append(np.nan)
            vs.append(np.nan)
    df_ps.loc[:, ['px_x']] = xs
    df_ps.loc[:, ['px_y']] = ys
    df_ps.loc[:, ['px_u']] = us
    df_ps.loc[:, ['px_v']] = vs
    df_ps.loc[:, ['timestamp']] = df_ps['TimeElapsed'] - df_ps['TimeElapsed'].iloc[0]
    df_ps.to_csv(f'data/P1S{sid}_vr.csv', index=False)
    
    # get the eye gaze in a frame
    df_vr = df_ps[['timestamp', 'px_x', 'px_y', 'px_u', 'px_v', 'Ped_Rotation_x', 'Ped_Rotation_y', 'Ped_Rotation_z']]
    df_meta = pd.read_csv(f'data/P1S{sid}_meta.csv', header=None)
    df_meta = df_meta.iloc[:, [5]]
    df_meta = df_meta.rename(columns={5: 'timestamp'})
    merged = pd.merge_asof(
        df_meta,
        df_vr,
        on='timestamp',
        direction='nearest'  # can also use 'backward' or 'forward'
    )

    # get the distance in the frame
    df_diff = df_ps[['timestamp']]
    df_diff.loc[:, ['dx']] = (df_ps['Ped_Location_x'] - df_ps['PodLeader_Location_x']) / 100 # turn to meters
    df_diff.loc[:, ['dy']] = (df_ps['Ped_Location_y'] - df_ps['PodLeader_Location_y']) / 100 # turn to meters
    df_diff_interp = df_diff.set_index('timestamp')
    df_diff_interp = pd.concat([df_diff_interp, pd.DataFrame(index=merged['timestamp'])])
    df_diff_interp = df_diff_interp[~df_diff_interp.index.duplicated(keep='first')]
    df_diff_interp.sort_index(inplace=True)
    df_diff_interp = df_diff_interp.interpolate(method='index')
    merged = pd.merge(merged, df_diff_interp.reset_index(), on='timestamp')

    merged.to_csv(f'data/P1S{sid}_merged.csv', index=False)


if __name__ == "__main__":
    # test_local()
    test_server()