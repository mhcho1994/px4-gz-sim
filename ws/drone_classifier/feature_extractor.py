import numpy as np
from pyulog import ULog
from pymavlink import mavutil

def extract_from_ulog(ulog_path):
    """ PX4 (.ulg) 파일에서 8개의 특징(Feature) 추출 """
    try:
        ulog = ULog(ulog_path)
        data = ulog.get_dataset('vehicle_local_position').data
        
        x, y, z = data['x'], data['y'], data['z']
        vx, vy, vz = data['vx'], data['vy'], data['vz']
        
        features = np.array([
            np.max(x) - np.min(x), np.max(y) - np.min(y), np.max(z) - np.min(z),
            np.std(vx), np.std(vy), np.std(vz),
            np.mean(z), np.max(np.sqrt(vx**2 + vy**2 + vz**2))
        ])
        return features
    except Exception as e:
        print(f"[경고] {ulog_path} 파싱 실패: {e}")
        return None

def extract_from_bin(bin_path):
    """ ArduPilot (.BIN) 파일에서 8개의 특징(Feature) 추출 """
    try:
        from pymavlink import mavutil
        mlog = mavutil.mavlink_connection(bin_path)
        
        x, y, z = [], [], []
        vx, vy, vz = [], [], []
        
        # 아듀파일럿 DataFlash 로그의 EKF 상태 데이터(XKF1 또는 NKF1)를 찾습니다.
        while True:
            msg = mlog.recv_match(type=['XKF1', 'NKF1'], blocking=False)
            if not msg:
                break
            
            # PN, PE, PD는 각각 북(North), 동(East), 하(Down) 방향의 위치(m)
            x.append(msg.PN)
            y.append(msg.PE)
            z.append(msg.PD)
            
            # VN, VE, VD는 각각 해당 방향의 속도(m/s)
            vx.append(msg.VN)
            vy.append(msg.VE)
            vz.append(msg.VD)
            
        if len(x) == 0:
            print(f"[경고] {bin_path}에 EKF 위치 데이터(XKF1/NKF1)가 없습니다.")
            return None
            
        x, y, z = np.array(x), np.array(y), np.array(z)
        vx, vy, vz = np.array(vx), np.array(vy), np.array(vz)
        
        # 앞서 PX4와 동일한 방식으로 8개의 특징값 계산
        features = np.array([
            np.max(x) - np.min(x), np.max(y) - np.min(y), np.max(z) - np.min(z),
            np.std(vx), np.std(vy), np.std(vz),
            np.mean(z), np.max(np.sqrt(vx**2 + vy**2 + vz**2))
        ])
        return features
    except Exception as e:
        print(f"[경고] {bin_path} 파싱 실패: {e}")
        return None