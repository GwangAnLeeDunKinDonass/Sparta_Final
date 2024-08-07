#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''
파일명: cleansing.py
작성자: 서민혁, 서영석
작성일자: 2024-07-22
수정 이력:
    - 2024-07-23 17:59 서영석
        * load_data 함수에 KDA 카운트 코드 추가
    - 2024-07-24 21:00 서영석
        * 병렬 처리를 위해 Dask를 사용하는 코드로 변환
    - 2024-07-25 13:00 서영석
        * 코드 간결화를 위해 지표 계산 함수 분리
    - 2024-07-26 13:00 서영석
        * 코드 간결화를 위해 calculate 함수들 calculate.py로 분리
    - 2024-08-06 14:00 서영석
        * SQL 정규화 된 데이터셋 최종데이터로 통합하는 함수 생성
설명: 데이터 정제를 위한 함수 모음
    - for_timeline : 매치 타임라인 데이터 추출 시,
    Output인 event와 participant 분할을 위해 사용
    - for_match_info : 경기 결과 데이터 추출 시,
    Output인 match_info 생성을 위해 사용
    - load_data : 정제 작업을 위해 추출된 Raw 데이터 Load 시 사용
    - load_final_data : SQL 정규화 된 데이터셋 최종데이터로 통합
'''
# Basic Library
import numpy as np
import pandas as pd
import json
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore', message='`meta` is not specified, inferred from partial data. Please provide `meta` if the result is unexpected.')
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

# Dask Library
import dask.dataframe as dd
from dask.diagnostics import ProgressBar

# Custormized Module
from .calculate import *

def for_timeline(df):
    # event용
    event_df = pd.DataFrame(df['events']) # 리스트를 DataFrame으로 변환
    event_df['match_id'] = df['match_id']  # match_id 열 추가

    # 분석에 필요한 행만 선택
    if 'type' in event_df.columns:
        event_df = event_df[event_df['type'].isin(['ITEM_PURCHASED', 'ITEM_UNDO',
                                                'CHAMPION_KILL', 'ITEM_SOLD',
                                                'BUILDING_KILL', 'ELITE_MONSTER_KILL'])].reset_index(drop=True)
    event_df = event_df.dropna(axis=1, how='all')

    # killerId와 creatorId는 행위자 Id로써
    # participantId와 동일한 성격을 띄므로 해당 컬럼 모두 participantId로 통일
    if 'participantId' not in event_df.columns:
        event_df['participantId'] = np.nan
    if 'creatorId' in event_df.columns:
        event_df.loc[event_df['creatorId'].notnull(), 'participantId'] = event_df['creatorId']
        event_df.drop('creatorId',axis=1,inplace=True)
    if 'killerId' in event_df.columns:    
        event_df.loc[event_df['killerId'].notnull(), 'participantId'] = event_df['killerId']
        event_df.drop('killerId',axis=1,inplace=True)
    
    # 분당 지표 생성을 위해 timestamp 분 단위 컬럼 생성
    if 'timestamp' in event_df.columns:    
        event_df.insert(1,'minute',0)
        event_df['minute'] = event_df['timestamp'].apply(lambda x: x // 60000)

    # assistingParticipantIds를 풀어 assist_1~4 컬럼으로 생성
    if 'assistingParticipantIds' in event_df.columns:
        assist = event_df[event_df['assistingParticipantIds'].notnull()][['type', 'match_id',
                                                                'minute', 'assistingParticipantIds', 
                                                                'participantId']]
        assist = assist[assist['participantId'] > 0]
        assist['assistingParticipantIds'] = assist['assistingParticipantIds'].apply(lambda x: list(set(x)))

        try:
            # # 람다 함수를 사용하여 assistingParticipantIds 필터링
            assist['assistingParticipantIds'] = assist.apply(
                lambda row: [aid for aid in row['assistingParticipantIds'] 
                            if (1 <= aid <= 5 and 1 <= row['participantId'] <= 5) 
                            or (6 <= aid <= 10 and 6 <= row['participantId'] <= 10) 
                            or (1 <= aid <= 10 and not (1 <= row['participantId'] <= 10))], axis=1)

            # assistingParticipantIds가 빈 리스트인 행 제거
            assist = assist[assist['assistingParticipantIds'].map(len) > 0]
            assist_list = assist['assistingParticipantIds'].apply(
                lambda x: pd.Series(x + [np.nan] * (4 - len(x))))
            assist[['a_1', 'a_2', 'a_3', 'a_4']] = assist_list

            event_df[['assist_1', 'assist_2', 'assist_3', 'assist_4']] = assist[['a_1', 'a_2', 'a_3', 'a_4']]

        except:
            pass
            
        event_df = event_df.drop('assistingParticipantIds',axis=1)

    # pf용
    p_df = pd.DataFrame(df['participantFrames']).T.reset_index(drop=True)
    p_df['timestamp'] = df['timestamp'] # timestamp열 추가
    p_df['timestamp'] = p_df['timestamp'].apply(lambda x: x // 60000)
    p_df['match_id'] = df['match_id']  # match_id 열 추가

    # damageStats' 열을 각각의 열로 분해하여 DataFrame으로 변환
    ds = pd.DataFrame(p_df['damageStats'].tolist())[['totalDamageDone','totalDamageDoneToChampions','totalDamageTaken']]
    # championStats 열을 각각의 열로 분해하여 DataFrame으로 변환
    cs = pd.DataFrame(p_df['championStats'].tolist())
    # position 열을 각각의 열로 분해하여 DataFrame으로 변환
    po = pd.DataFrame(p_df['position'].tolist())

    # position DataFrame 생성
    position_df = p_df[['match_id', 'participantId', 'timestamp']].copy()
    position_df[['position_x', 'position_y']] = po

    # 원래의 participant DataFrame과 분해된 DataFrame들을 열기준으로 결합
    participant_df = pd.concat([p_df, ds, cs], axis=1)
    # 불필요한 열 제거
    participant_df = participant_df.drop(['championStats', 'damageStats', 'position'], axis=1)

    participant_df = pd.concat([participant_df.iloc[:, 9],
                                participant_df.iloc[:, :9],
                                participant_df.iloc[:, 10:]], axis=1)
    participant_df = pd.concat([participant_df.iloc[:, 10],
                                participant_df.iloc[:, :10],
                                participant_df.iloc[:, 11:]], axis=1)
    
    position_df['lane'] = position_df.apply(lambda row: calculate_lane(row['position_x'], row['position_y']), axis=1)

    return event_df, participant_df, position_df

# -------------------------------------------------------------------------------

def for_match_info(json_file):
    match_df = pd.DataFrame(json_file['participants'])

    required_columns = ['teamId', 'puuid', 'summonerName', 'summonerId', 'participantId',
                        'teamPosition', 'challenges', 'championName', 'lane',
                        'kills', 'deaths', 'assists', 'summoner1Id', 'summoner2Id',
                        'totalMinionsKilled', 'neutralMinionsKilled', 'goldEarned', 'goldSpent',
                        'champExperience', 'item0', 'item1', 'item2', 'item3', 'item4', 'item5',
                        'item6', 'totalDamageDealt', 'totalDamageDealtToChampions', 'totalDamageTaken',
                        'damageDealtToBuildings', 'damageDealtToObjectives', 'damageDealtToTurrets',
                        'totalTimeSpentDead', 'visionScore', 'win', 'timePlayed']

    for col in required_columns:
        if col not in match_df.columns:
            match_df[col] = np.nan

    sample = match_df[required_columns]

    if 'challenges' in sample.columns:
        challenge = pd.DataFrame(sample['challenges'].tolist())

        challenge_columns = ['abilityUses', 'skillshotsDodged', 'skillshotsHit',
                             'enemyChampionImmobilizations', 'laneMinionsFirst10Minutes',
                             'controlWardsPlaced', 'wardTakedowns', 'effectiveHealAndShielding',
                             'dragonTakedowns', 'baronTakedowns']

        jungle_col = challenge.filter(regex='^jungle|Jungle|kda')

        for col in challenge_columns:
            if col not in challenge.columns:
                challenge[col] = np.nan

        for col in jungle_col.columns:
            if col not in challenge.columns:
                challenge[col] = np.nan

        match_info = pd.concat([sample, challenge[challenge_columns], jungle_col], axis=1)
        match_info = match_info.drop(['challenges'], axis=1)

        if 'moreEnemyJungleThanOpponent' in match_info.columns:
            match_info = match_info.drop(['moreEnemyJungleThanOpponent'], axis=1)

    else:
        match_info = sample

    if match_info.empty or match_info.isna().all().all():
        return None

    # 10줄이 아닌 경우 None 반환
    if len(match_info) != 10:
        return None

    # ---------------------------------------------------------------------------
    object_df = pd.DataFrame(json_file['teams'])
    objectives = pd.json_normalize(object_df['objectives'])

    # 추출할 오브젝트의 목록 정의
    possible_objectives = ['baron.kills', 'champion.kills', 'dragon.kills', 'horde.kills', 'inhibitor.kills', 'riftHerald.kills', 'tower.kills']

    # 실제로 존재하는 오브젝트만 추출
    existing_objectives = [col for col in possible_objectives if col in objectives.columns]

    # 각 오브젝트의 kills 값만 추출하고 열 이름 변경
    objectives = objectives[existing_objectives]
    objectives.columns = [col.split('.')[0] + '_kills' for col in objectives.columns]

    # teamId와 결합하여 최종 데이터프레임 생성
    objectives = pd.concat([object_df[['teamId']], objectives], axis=1)

    return match_info, objectives
    
# -------------------------------------------------------------------------------    

def load_data(directory='./data', to_pandas=True):
    data_list = ['event', 'participant', 'position', 'match_info','objectives']
    df_list = []
    print('데이터 불러오는 중..')
    for dt in data_list:
        print(dt)
        dt_pocket = []
        for i in tqdm(range(1,len(os.listdir(f'{directory}/{dt}'))+1)):
            file = dd.read_csv(
                f'{directory}/{dt}/{dt}{i}.csv',
                dtype={'buildingType': 'object', 'killType': 'object', 
                       'laneType': 'object', 'monsterSubType': 'object',
                       'monsterType': 'object', 'position': 'object',
                       'towerType': 'object', 'transformType': 'object',
                       'victimDamageDealt': 'object', 'victimDamageReceived': 'object',
                       'wardType': 'object', 'levelUpType' : 'object'}
            )
            dt_pocket.append(file)
        df = dd.concat(dt_pocket, axis=0)
        df_list.append(df)

    print('데이터 정제 중..')
    event = df_list[0]
    participant = df_list[1]
    position = df_list[2]
    match_info = df_list[3]
    objectives = df_list[4]

    participant = dd.merge(participant, 
                           match_info[['match_id', 'puuid', 'participantId', 'teamPosition', 'championName']], 
                           on=['match_id', 'participantId'], 
                           how='inner')
    
    cols = participant.columns.tolist()
    cols.insert(0, cols.pop(cols.index('puuid')))
    cols.insert(4, cols.pop(cols.index('teamPosition')))
    cols.insert(5, cols.pop(cols.index('championName')))
    participant = participant[cols]

    print('지표 추가 중..')
    # with ProgressBar():
        # print('초기 세팅 중..')
        # event = event.persist()
    with ProgressBar():
        print('KDA')
        participant = participant.persist()
        participant = calculate_KDA(participant, event)
    with ProgressBar():
        print('Involve')
        participant = participant.persist()
        participant = calculate_involve(participant, event)
    with ProgressBar():
        print('Item')
        participant = participant.persist()
        participant = calculate_item(participant, event)
    # with ProgressBar():
    #     print('Position Change')
    #     position = position.persist()
    #     position = calculate_position_change(position)
    with ProgressBar():
        print('캐시 저장 및 중복 제거 중..')
        participant = participant.persist()
        # position = position.persist()

    # event = event.drop_duplicates(subset=['match_id', 'timestamp', 'participantId'])
    participant = participant.drop_duplicates(subset=['puuid', 'match_id', 'timestamp', 'participantId'])
    position = position.drop_duplicates(subset=['match_id', 'participantId', 'timestamp'])
    match_info = match_info.drop_duplicates(subset=['puuid', 'match_id', 'participantId'])
    objectives = objectives.drop_duplicates(subset=['match_id', 'teamId'])

    # position = position.reset_index(drop=True)
    # if position.index.duplicated().any():
    #     print("중복된 인덱스가 발견되었습니다. 중복된 인덱스를 제거합니다.")
    #     position = position[~position.index.duplicated(keep='first')]

    if to_pandas:
        print('DASK -> Pandas로 변환 중')
        with ProgressBar():
            # print('event')
            # event = event.compute().sort_values(['match_id','participantId','timestamp']).reset_index(drop=True)
            print('participant')
            participant = participant.compute().sort_values(['puuid','match_id','timestamp']).reset_index(drop=True)
            print('position')
            position = position.compute().sort_values(['match_id','participantId','timestamp']).reset_index(drop=True)
            print('match_info')
            match_info = match_info.compute().sort_values(['match_id','participantId']).reset_index(drop=True)
            print('objectives')
            objectives = objectives.compute().sort_values(['match_id','teamId']).reset_index(drop=True)

    print('완료')
    
    return participant, position, match_info, objectives

# -------------------------------------------------------------------------------    

def load_final_data(directory=('./data/after_norm')):
    csv_list = os.listdir('./data/after_norm')
    
    print('데이터 불러오는 중..')
    champion = pd.read_csv(os.path.join(directory,csv_list[0]))
    match_tier = pd.read_csv(os.path.join(directory,csv_list[1]))
    participant_minute = pd.read_csv(os.path.join(directory,csv_list[2]))
    participant_preset = pd.read_csv(os.path.join(directory,csv_list[3]))
    participant_result = pd.read_csv(os.path.join(directory,csv_list[4]))
    team_result = pd.read_csv(os.path.join(directory,csv_list[5]))
    
    df = pd.merge(match_tier,team_result,
                  on=['match_id'])
    df = pd.merge(df,participant_preset,
                  on=['match_id','teamId'])
    df = pd.merge(df, participant_result, 
                  on=['match_id','puuid'])
    df = pd.merge(df, champion,
                  on=['championName'])
    
    df['subRole'] = df['subRole'].fillna('-')

    column_list = [col for col in participant_minute.drop(['match_id','puuid','timestamp','lane'],axis=1)]
    print('데이터 병합 중..')
    for col in tqdm(column_list):
        if col==column_list[0]:
            # 타워 관여율
            involve = participant_minute.groupby(['match_id',
                                                  'puuid'],as_index=False)[['involve_tower']].sum()

            result = pd.merge(df, involve[['puuid','match_id','involve_tower']],
                              on=['puuid','match_id'],
                              how='inner')

            result['tower_involve_ratio'] = round(result['involve_tower']/result['building_kills']*100,1)
            result['tower_involve_ratio'] = result['tower_involve_ratio'].fillna(0)
            result = result.drop(['involve_tower'],axis=1)

        elif col==column_list[1]:
            # 오브젝트 관여율
            involve = participant_minute.groupby(['match_id',
                                                  'puuid'],as_index=False)[['involve_object']].sum()

            result = pd.merge(result, involve[['puuid','match_id','involve_object']],
                              on=['puuid','match_id'],
                              how='inner')

            result['object_involve_ratio'] = round(result['involve_object']/result['monster_kills']*100,1)
            result['object_involve_ratio'] = result['object_involve_ratio'].fillna(0)
            result = result.drop(['involve_object'],axis=1)

        elif col==column_list[2]:
            # 10분당 순거래율
            summary = participant_minute.groupby(['match_id', 'puuid']).agg({
                'transaction_margin': 'sum',
                'timestamp': 'max'
            }).reset_index()

            summary['average_transaction_margin_per_10min'] = round(summary['transaction_margin'] / (summary['timestamp'] / 10),2)

            result = pd.merge(result,
                              summary[['match_id','puuid','average_transaction_margin_per_10min']],
                              on = ['match_id','puuid'],
                              how = 'inner')

        elif col in column_list[3:7]:
            # 스탯
            stat = participant_minute[['match_id','puuid','timestamp', col]]
            stat[col] = stat.groupby(['match_id', 'puuid'])[col].diff()
            mean_changes = stat.groupby(['match_id', 'puuid'])[col].mean().reset_index()
            mean_changes[col] = mean_changes[col].round(1)

            result = pd.merge(result,mean_changes,
                              on = ['match_id','puuid'],
                              how = 'inner')
            
            result = result.rename(columns={col:f'delta_{col}'})

        elif col==column_list[7]:
            # 안움직임
            po_anorm = participant_minute[['match_id','timestamp','puuid','lane','position_change']]
            po_anorm = po_anorm[po_anorm['timestamp']>0]
            po_anorm = po_anorm[po_anorm['position_change']==0]
            po_anorm = po_anorm[po_anorm['lane'].isin(['red_zone','blue_zone'])]

            po_anorm = po_anorm.groupby(['match_id', 'puuid']).size().reset_index(name='no_moving_minute')

            result = pd.merge(result,po_anorm,
                              on = ['match_id', 'puuid'],
                              how = 'left')

            result['no_moving_minute'] = result['no_moving_minute'].fillna(0)

        elif col==column_list[8]:
            # 10분전 데스
            filtered_pa = participant_minute[participant_minute['timestamp'] <= 10]
            summary = filtered_pa.groupby(['match_id', 'puuid'],as_index=False)[['death']].sum()
            summary = summary.rename(columns={'death':'deathBefore10Minutes'})

            result = pd.merge(result,summary,
                              on = ['match_id', 'puuid'],
                              how = 'left')
            
        elif col==column_list[9]:
            # 10분전 누적 골드
            filtered_pa = participant_minute[['match_id', 'puuid', col]]
            filtered_pa = filtered_pa[participant_minute['timestamp'] == 10]
            filtered_pa = filtered_pa.rename(columns={'totalGold':'goldBefore10Minutes'})
            
            result = pd.merge(result,filtered_pa,
                              on = ['match_id', 'puuid'],
                              how = 'left')
            
        elif col==column_list[10]:
            # 10분 레벨
            filtered_pa = participant_minute[['match_id', 'puuid', col]]
            filtered_pa = filtered_pa[participant_minute['timestamp'] == 10]
            filtered_pa = filtered_pa.rename(columns={'level':'level10Minutes'})

            result = pd.merge(result,filtered_pa,
                              on = ['match_id', 'puuid'],
                              how = 'left')
            
    result = pd.concat([result.iloc[:,:11],
                        result.iloc[:,31:33],
                        result.iloc[:,11:31],
                        result.iloc[:,33:]], axis=1)

    # 소환사 주문 딕셔너리
    summoner_spells = {
        1: "Cleanse",
        3: "Exhaust",
        4: "Flash",
        6: "Ghost",
        7: "Heal",
        11: "Smite",
        12: "Teleport",
        14: "Ignite",
        21: "Barrier",
        30: "To the King!",
        31: "Poro Toss",
        32: "Mark"
    }

    # summoner1Id 및 summoner2Id 열을 소환사 주문 이름으로 변환
    result['summoner1Id'] = result['summoner1Id'].map(summoner_spells)
    result['summoner2Id'] = result['summoner2Id'].map(summoner_spells)

    return result

