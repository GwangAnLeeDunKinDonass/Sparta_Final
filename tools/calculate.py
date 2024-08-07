'''
파일명: calculate.py
작성자: 서민혁, 서영석
작성일자: 2024-07-26
수정 이력:
설명: 지표 계산을 위한 함수 모음 (편의상 calcuate 생략)
    - lane : 해당 위치에 맞는 라인명 반환
    - position_change : 1분전과 위상변화값 반환
    - KDA : 킬, 데스, 어시스트 개수 반환
    - Involve : 타워 철거, 오브젝트 처치 관여 수 반환
    - Item : 순거래량 반환
'''

# Basic Library
import numpy as np
import pandas as pd

# Dask Library
from dask.diagnostics import ProgressBar
import dask.dataframe as dd

def calculate_lane(x, y):
    top_ranges = [(500, 2000, 6000, 14000), (600, 9000, 13000, 14500), (1900, 4500, 11100, 13100)]
    bottom_ranges = [(6000, 14000, 500, 2000), (13000, 14500, 500, 9000), (10500, 13000, 2000, 3800)]
    mid_ranges = [(4500, 6000, 4500, 6000), (5200, 6700, 5200, 6700), (5900, 7400, 5900, 7400), (6000, 8500, 6000, 8500),
                  (7300, 8800, 7300, 8800), (8000, 9500, 8000, 9500), (8700, 10200, 8700, 10200), (9200, 10500, 9200, 10500)]
    blue_zone = [(0, 4500, 0, 4500)]
    red_zone = [(10500, 15000, 10500, 15000)]

    for range_ in top_ranges:
        if range_[0] <= x <= range_[1] and range_[2] <= y <= range_[3]:
            return 'top'
    for range_ in mid_ranges:
        if range_[0] <= x <= range_[1] and range_[2] <= y <= range_[3]:
            return 'mid'
    for range_ in bottom_ranges:
        if range_[0] <= x <= range_[1] and range_[2] <= y <= range_[3]:
            return 'bottom'
    for range_ in blue_zone:
        if range_[0] <= x <= range_[1] and range_[2] <= y <= range_[3]:
            return 'blue_zone'
    for range_ in red_zone:
        if range_[0] <= x <= range_[1] and range_[2] <= y <= range_[3]:
            return 'red_zone'
    return 'jungle'  # 나머지는 jungle

# -------------------------------------------------------------------------------

def calculate_position_change(df):
    df = df.sort_values(by=['match_id', 'participantId', 'timestamp'])
    df['prev_position_x'] = df.groupby(['match_id', 'participantId'])['position_x'].shift(1)
    df['prev_position_y'] = df.groupby(['match_id', 'participantId'])['position_y'].shift(1)

    df['position_change'] = np.sqrt((df['position_x'] - df['prev_position_x']) ** 2 + (df['position_y'] - df['prev_position_y']) ** 2)
    df['position_change'] = df['position_change'].fillna(0)
    df = df.drop(columns=['prev_position_x', 'prev_position_y'])
    return df

# -------------------------------------------------------------------------------

def calculate_KDA(participant, event):
    with ProgressBar():
        # 킬
        kill_cnt = event[event['type'] == 'CHAMPION_KILL']
        kill_cnt = kill_cnt.groupby(['minute', 'match_id', 'participantId']).size().reset_index()
        kill_cnt = kill_cnt.rename(columns={0: 'kill', 'minute': 'timestamp'})

        # 데스
        death_cnt = event[event['type'] == 'CHAMPION_KILL']
        death_cnt = death_cnt.groupby(['minute', 'match_id', 'victimId']).size().reset_index()
        death_cnt = death_cnt.rename(columns={'victimId': 'participantId', 0: 'death', 'minute': 'timestamp'})

        # 어시스트
        dfs = []
        for col in ['assist_1', 'assist_2', 'assist_3', 'assist_4']:
            df_temp = event[event['type'] == 'CHAMPION_KILL']
            df_temp = df_temp.groupby(['type', 'match_id', 'minute', col]).size().reset_index()
            df_temp = df_temp.rename(columns={col: 'participantId', 0: 'assist'})
            dfs.append(df_temp)

        t_df = dd.concat(dfs, axis=0)

        assist_cnt = t_df.groupby(['type', 'match_id', 'minute', 'participantId'])['assist'].sum().reset_index()
        assist_cnt = assist_cnt.drop('type', axis=1)
        assist_cnt = assist_cnt.rename(columns={'minute': 'timestamp'})

        # KDA 통합
        kill_cnt['timestamp'] = kill_cnt['timestamp'].astype('int64')
        kill_cnt['participantId'] = kill_cnt['participantId'].astype('int64')
        death_cnt['timestamp'] = death_cnt['timestamp'].astype('int64')
        death_cnt['participantId'] = death_cnt['participantId'].astype('int64')
        assist_cnt['timestamp'] = assist_cnt['timestamp'].astype('int64')
        assist_cnt['participantId'] = assist_cnt['participantId'].astype('int64')

        participant = dd.merge(participant, kill_cnt, 
                            on=['timestamp', 'match_id', 'participantId'], 
                            how='left')
        participant = dd.merge(participant, death_cnt, 
                            on=['timestamp', 'match_id', 'participantId'], 
                            how='left')
        participant = dd.merge(participant, assist_cnt, 
                            on=['timestamp', 'match_id', 'participantId'], 
                            how='left')

        participant['kill'] = participant['kill'].fillna(0)
        participant['death'] = participant['death'].fillna(0)
        participant['assist'] = participant['assist'].fillna(0)
        participant[['kill', 'death', 'assist']] = participant[['kill', 'death', 'assist']].astype('int64')

        return participant

# -------------------------------------------------------------------------------

def calculate_involve(participant, event):
    with ProgressBar():
        # 타워 철거 관여
        dfs = []
        for col in ['assist_1', 'assist_2', 'assist_3', 'assist_4']:
            df_temp = event[event['type'] == 'BUILDING_KILL']
            df_temp = df_temp.groupby(['type', 'match_id', 'minute', col]).size().reset_index()
            df_temp = df_temp.rename(columns={col: 'participantId', 0: 'tower_assist'})
            dfs.append(df_temp)

        tower_a = dd.concat(dfs, axis=0)

        tower_k = event[event['type'] == 'BUILDING_KILL'].groupby(['type', 'match_id', 'minute', 'participantId']).size().reset_index()
        tower_k = tower_k.rename(columns={0: 'tower_kill'})
        final_tower = dd.merge(tower_k, tower_a, on=['type', 'match_id', 'minute', 'participantId'], how='left')
        final_tower['tower_assist'] = final_tower['tower_assist'].fillna(0)
        final_tower['involve_tower'] = final_tower['tower_kill'] + final_tower['tower_assist']
        final_tower = final_tower.rename(columns={'minute': 'timestamp'})
        final_tower = final_tower.drop(['type', 'tower_kill', 'tower_assist'], axis=1)

        # 오브젝트 처치 관여
        dfs = []
        for col in ['assist_1', 'assist_2', 'assist_3', 'assist_4']:
            df_temp = event[event['type'] == 'ELITE_MONSTER_KILL']
            df_temp = df_temp.groupby(['type', 'match_id', 'minute', col]).size().reset_index()
            df_temp = df_temp.rename(columns={col: 'participantId', 0: 'object_assist'})
            dfs.append(df_temp)

        monster_a = dd.concat(dfs, axis=0)

        monster_k = event[event['type'] == 'ELITE_MONSTER_KILL'].groupby(['type', 'match_id', 'minute', 'participantId']).size().reset_index()
        monster_k = monster_k.rename(columns={0: 'object_kill'})
        final_monster = dd.merge(monster_k, monster_a, on=['type', 'match_id', 'minute', 'participantId'], how='left')
        final_monster['object_assist'] = final_monster['object_assist'].fillna(0)
        final_monster['involve_object'] = final_monster['object_kill'] + final_monster['object_assist']
        final_monster = final_monster.rename(columns={'minute': 'timestamp'})
        final_monster = final_monster.drop(['type', 'object_kill', 'object_assist'], axis=1)

        # 타워, 오브젝트 통합
        final_tower['timestamp'] = final_tower['timestamp'].astype('int64')
        final_tower['participantId'] = final_tower['participantId'].astype('int64')
        final_monster['timestamp'] = final_monster['timestamp'].astype('int64')
        final_monster['participantId'] = final_monster['participantId'].astype('int64')

        participant = dd.merge(participant, final_tower,
                            on=['timestamp', 'match_id', 'participantId'], 
                            how='left')
        participant = dd.merge(participant, final_monster,
                            on=['timestamp', 'match_id', 'participantId'], 
                            how='left')

        participant['involve_tower'] = participant['involve_tower'].fillna(0)
        participant['involve_object'] = participant['involve_object'].fillna(0)

        participant[['involve_tower', 'involve_object']] = participant[['involve_tower', 'involve_object']].astype('int')

        return participant

# -------------------------------------------------------------------------------

def calculate_item(participant, event):
    # 아이템 판매
    sell_df = event[event['type'] == 'ITEM_SOLD']
    sell_df = sell_df.groupby(['match_id', 'minute', 'participantId']).size().reset_index()
    sell_df = sell_df.rename(columns={0: 'item_sell', 'minute': 'timestamp'})

    # 아이템 구매
    buy_df = event[event['type'] == 'ITEM_PURCHASED']
    buy_df = buy_df.groupby(['match_id', 'minute', 'participantId']).size().reset_index()
    buy_df = buy_df.rename(columns={0: 'item_buy', 'minute': 'timestamp'})

    # 아이템 되돌리기
    undo_df = event[event['type'] == 'ITEM_UNDO']
    undo_df = undo_df.groupby(['match_id', 'minute', 'participantId']).size().reset_index()
    undo_df = undo_df.rename(columns={0: 'item_undo', 'minute': 'timestamp'})

    # 데이터 병합
    tran_df = dd.merge(buy_df, sell_df, on=['match_id', 'timestamp', 'participantId'], how='outer')
    tran_df = dd.merge(tran_df, undo_df, on=['match_id', 'timestamp', 'participantId'], how='outer')

    tran_df['timestamp'] = tran_df['timestamp'].astype('int64')
    tran_df['participantId'] = tran_df['participantId'].astype('int64')

    tran_df['item_sell'] = tran_df['item_sell'].fillna(0)
    tran_df['item_buy'] = tran_df['item_buy'].fillna(0)
    tran_df['item_undo'] = tran_df['item_undo'].fillna(0)

    tran_df['item_transaction'] = tran_df['item_sell'] + tran_df['item_buy']

    tran_df['item_transaction'] = tran_df.apply(
        lambda row: 1 if row['timestamp'] == 0 and row['item_transaction'] > 1 else row['item_transaction'],
        axis=1, meta=('item_transaction', 'int64')
    )

    tran_df = tran_df.drop(['item_sell', 'item_buy'], axis=1)
    participant = dd.merge(participant, tran_df, on=['match_id', 'timestamp', 'participantId'], how='left')
    participant[['item_transaction', 'item_undo']] = participant[['item_transaction', 'item_undo']].fillna(0)
    participant[['item_transaction', 'item_undo']] = participant[['item_transaction', 'item_undo']].astype('int64')

    return participant
