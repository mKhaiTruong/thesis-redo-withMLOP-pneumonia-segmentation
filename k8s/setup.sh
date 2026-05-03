#!/bin/bash
echo "=== Creating PVC ==="
kubectl apply -f k8s/pvc.yaml

echo "=== Creating secrets ==="
kubectl create secret generic app-secrets --from-env-file=.env --dry-run=client -o yaml | kubectl apply -f -

echo "=== Applying services ==="
kubectl apply -f k8s/services/
kubectl apply -f k8s/prometheus-config.yaml

echo "=== Restarting deployments ==="
kubectl rollout restart deployment/ingestion
kubectl rollout restart deployment/transformation
kubectl rollout restart deployment/data-drift
kubectl rollout restart deployment/orchestrator
kubectl rollout restart deployment/lstm
kubectl rollout restart deployment/dqn
kubectl rollout restart deployment/claude_validation
kubectl rollout restart deployment/ai_manager

echo "=== Pods status ==="
kubectl get pods -w