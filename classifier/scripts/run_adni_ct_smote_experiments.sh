#!/bin/bash

# ADNI_CT SMOTE 병렬 실험 스크립트
# 여러 GPU를 사용한 병렬 실행

echo "=========================================="
echo "ADNI_CT SMOTE 병렬 실험 시작"
echo "총 14개 실험: MLP(2개) + MLP-A/GCN/GAT(각 4개)"
echo "=========================================="

# 실험 설정
DATA="adni_ct"
EPOCHS=5000
SEED=0

# 사용 가능한 GPU 리스트 (4개 GPU 사용)
GPUS=(4 5 6 7)
GPU_COUNT=${#GPUS[@]}

# 모델 리스트
MODELS_WITHOUT_ADJ=("mlp")  # adjacency matrix 사용하지 않는 모델
MODELS_WITH_ADJ=("mlp-a" "gcn" "gat")  # adjacency matrix 사용하는 모델

# Augmentation 설정
AUG_LEVELS=("min" "full")
ADJ_ASSIGNMENTS=("random" "average")

# 실험 카운터
experiment_count=0
total_experiments=14  # MLP: 2개 (min/full), MLP-A/GCN/GAT: 각각 4개 (min/full × random/average) = 2 + 3*4 = 14
gpu_index=0

# 실험 시작 시간
start_time=$(date)
echo "실험 시작 시간: $start_time"
echo "사용 GPU: ${GPUS[*]}"
echo "최대 병렬 실행: $GPU_COUNT"
echo ""

# 실험 로그 디렉토리 생성
LOG_DIR="./parallel_experiment_logs"
mkdir -p $LOG_DIR

# 백그라운드 프로세스 PID 저장 배열
declare -a bg_pids=()

# MLP 모델 실험 (adjacency matrix 사용 안 함)
for model in "${MODELS_WITHOUT_ADJ[@]}"; do
    for aug_level in "${AUG_LEVELS[@]}"; do
        experiment_count=$((experiment_count + 1))
        
        # GPU 할당 (라운드 로빈 방식)
        current_gpu=${GPUS[$gpu_index]}
        gpu_index=$(( (gpu_index + 1) % GPU_COUNT ))
        
        # 실험 이름 생성
        exp_name="${model}_SMOTE_${aug_level}"
        log_file="$LOG_DIR/${exp_name}.log"
        
        echo "[$experiment_count/$total_experiments] 실험 시작: $exp_name (GPU: $current_gpu)"
        
        # 백그라운드에서 실험 실행
        (
            echo "=== 실험 시작: $exp_name ===" > $log_file
            echo "GPU: $current_gpu" >> $log_file
            echo "시작 시간: $(date)" >> $log_file
            echo "" >> $log_file
            
            python main.py \
                --data $DATA \
                --model $model \
                --epochs $EPOCHS \
                --device $current_gpu \
                --seed_num $SEED \
                --augmentation SMOTE \
                --aug_level $aug_level >> $log_file 2>&1
            
            exit_code=$?
            echo "" >> $log_file
            echo "종료 시간: $(date)" >> $log_file
            echo "종료 코드: $exit_code" >> $log_file
            
            if [ $exit_code -eq 0 ]; then
                echo "✓ 실험 완료: $exp_name" >> $log_file
                echo "✓ 실험 완료: $exp_name"
            else
                echo "✗ 실험 실패: $exp_name (코드: $exit_code)" >> $log_file
                echo "✗ 실험 실패: $exp_name (코드: $exit_code)"
            fi
        ) &
        
        # 백그라운드 프로세스 PID 저장
        bg_pids+=($!)
        
        # GPU 수만큼 프로세스가 실행 중이면 대기
        if [ ${#bg_pids[@]} -ge $GPU_COUNT ]; then
            # 하나의 프로세스가 완료될 때까지 대기
            wait ${bg_pids[0]}
            # 완료된 프로세스 PID 제거
            bg_pids=("${bg_pids[@]:1}")
        fi
        
        # 짧은 대기 시간 (GPU 메모리 정리)
        sleep 2
    done
done

# MLP-A, GCN, GAT 모델 실험 (adjacency matrix 사용)
for model in "${MODELS_WITH_ADJ[@]}"; do
    for aug_level in "${AUG_LEVELS[@]}"; do
        for adj_assignment in "${ADJ_ASSIGNMENTS[@]}"; do
            experiment_count=$((experiment_count + 1))
            
            # GPU 할당 (라운드 로빈 방식)
            current_gpu=${GPUS[$gpu_index]}
            gpu_index=$(( (gpu_index + 1) % GPU_COUNT ))
            
            # 실험 이름 생성
            exp_name="${model}_SMOTE_${aug_level}_${adj_assignment}"
            log_file="$LOG_DIR/${exp_name}.log"
            
            echo "[$experiment_count/$total_experiments] 실험 시작: $exp_name (GPU: $current_gpu)"
            
            # 백그라운드에서 실험 실행
            (
                echo "=== 실험 시작: $exp_name ===" > $log_file
                echo "GPU: $current_gpu" >> $log_file
                echo "시작 시간: $(date)" >> $log_file
                echo "" >> $log_file
                
                python main.py \
                    --data $DATA \
                    --model $model \
                    --epochs $EPOCHS \
                    --device $current_gpu \
                    --seed_num $SEED \
                    --augmentation SMOTE \
                    --aug_level $aug_level \
                    --adj_assignment $adj_assignment \
                    --adj_percentile 90 >> $log_file 2>&1
                
                exit_code=$?
                echo "" >> $log_file
                echo "종료 시간: $(date)" >> $log_file
                echo "종료 코드: $exit_code" >> $log_file
                
                if [ $exit_code -eq 0 ]; then
                    echo "✓ 실험 완료: $exp_name" >> $log_file
                    echo "✓ 실험 완료: $exp_name"
                else
                    echo "✗ 실험 실패: $exp_name (코드: $exit_code)" >> $log_file
                    echo "✗ 실험 실패: $exp_name (코드: $exit_code)"
                fi
            ) &
            
            # 백그라운드 프로세스 PID 저장
            bg_pids+=($!)
            
            # GPU 수만큼 프로세스가 실행 중이면 대기
            if [ ${#bg_pids[@]} -ge $GPU_COUNT ]; then
                # 하나의 프로세스가 완료될 때까지 대기
                wait ${bg_pids[0]}
                # 완료된 프로세스 PID 제거
                bg_pids=("${bg_pids[@]:1}")
            fi
            
            # 짧은 대기 시간 (GPU 메모리 정리)
            sleep 2
        done
    done
done

# 모든 백그라운드 프로세스 완료 대기
echo ""
echo "모든 실험 시작 완료. 백그라운드 프로세스 완료 대기 중..."
echo "진행 상황은 다음 명령어로 확인할 수 있습니다:"
echo "  watch -n 5 'ls -la $LOG_DIR/*.log | wc -l; echo \"완료된 로그 파일 수 (총 $total_experiments개)\"'"
echo ""

# 남은 모든 백그라운드 프로세스 완료 대기
for pid in "${bg_pids[@]}"; do
    wait $pid
done

# 실험 완료 시간
end_time=$(date)
echo "=========================================="
echo "모든 실험 완료!"
echo "시작 시간: $start_time"
echo "종료 시간: $end_time"
echo "총 실험 수: $total_experiments"
echo "=========================================="

# 결과 요약
echo ""
echo "실험 결과 요약:"
echo "----------------------------------------"
success_count=0
fail_count=0

# MLP 모델 결과 확인
for model in "${MODELS_WITHOUT_ADJ[@]}"; do
    for aug_level in "${AUG_LEVELS[@]}"; do
        exp_name="${model}_SMOTE_${aug_level}"
        log_file="$LOG_DIR/${exp_name}.log"
        
        if [ -f "$log_file" ]; then
            if grep -q "✓ 실험 완료" "$log_file"; then
                echo "✓ $exp_name"
                success_count=$((success_count + 1))
            else
                echo "✗ $exp_name"
                fail_count=$((fail_count + 1))
            fi
        else
            echo "? $exp_name (로그 파일 없음)"
            fail_count=$((fail_count + 1))
        fi
    done
done

# MLP-A, GCN, GAT 모델 결과 확인
for model in "${MODELS_WITH_ADJ[@]}"; do
    for aug_level in "${AUG_LEVELS[@]}"; do
        for adj_assignment in "${ADJ_ASSIGNMENTS[@]}"; do
            exp_name="${model}_SMOTE_${aug_level}_${adj_assignment}"
            log_file="$LOG_DIR/${exp_name}.log"
            
            if [ -f "$log_file" ]; then
                if grep -q "✓ 실험 완료" "$log_file"; then
                    echo "✓ $exp_name"
                    success_count=$((success_count + 1))
                else
                    echo "✗ $exp_name"
                    fail_count=$((fail_count + 1))
                fi
            else
                echo "? $exp_name (로그 파일 없음)"
                fail_count=$((fail_count + 1))
            fi
        done
    done
done

echo "----------------------------------------"
echo "성공: $success_count/$total_experiments"
echo "실패: $fail_count/$total_experiments"

# 결과 파일 확인
echo ""
echo "결과 파일 확인:"
if [ -f "/home/user14/bagle/summary/experiment_summary.xlsx" ]; then
    echo "✓ 실험 결과가 저장되었습니다: /home/user14/bagle/summary/experiment_summary.xlsx"
else
    echo "✗ 실험 결과 파일을 찾을 수 없습니다."
fi

echo ""
echo "상세 로그는 다음 디렉토리에서 확인할 수 있습니다:"
echo "  $LOG_DIR/"
echo ""
echo "예: cat $LOG_DIR/gcn_SMOTE_min_average.log"
