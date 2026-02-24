# COMPREHENSIVE PRODUCT REQUIREMENTS DOCUMENT (PRD)
## AI-Powered Root Cause Analysis (RCA) System for Production Failures

---

## EXECUTIVE SUMMARY

### Product Vision
An intelligent, AI-powered system that automatically diagnoses root causes of production failures in complex distributed systems by analyzing system metrics, logs, and system dependencies through advanced machine learning, statistical causal inference, and graph-based reasoning.

### Problem Statement
- **Traditional Approach**: Engineers manually spend 2-8 hours correlating logs, metrics, and deployment events to identify root causes
- **Current Limitations**: Static thresholds generate false positives; rule-based systems can't adapt to novel failure patterns; purely data-driven approaches can't distinguish correlation from causation
- **Business Impact**: Every minute of downtime in production systems costs organizations thousands to millions of dollars in lost revenue

### Solution Overview
The RCA System automates this process through a 4-stage pipeline:
1. **Normal Behavior Learning**: LSTM autoencoders trained on healthy operational data
2. **Anomaly Detection**: Identifies deviations from normal patterns
3. **Causal Inference**: Uses Granger causality and graph algorithms to determine true causes
4. **Root Cause Ranking**: Scores and ranks candidates with confidence percentages

### Key Value Propositions
- ✅ **MTTR Reduction**: From hours to 15-30 minutes
- ✅ **Accuracy**: 70%+ top-1 accuracy, 85%+ top-3 accuracy
- ✅ **Coverage**: Detects 50+ distinct anomaly types
- ✅ **Transparency**: Provides causal chains and supporting evidence
- ✅ **Scalability**: Handles 100+ metrics from cloud platforms

---

## 1. DETAILED FUNCTIONAL REQUIREMENTS

### 1.1 Core Capabilities (What the System CAN Do)

#### **1.1.1 Data Ingestion & Integration**

**Requirement**: System must seamlessly integrate with production monitoring infrastructure

**Scope**:
```
INPUT DATA SOURCES:
├── Prometheus
│   ├── Node exporter metrics (CPU, memory, disk, network)
│   ├── Application metrics (latency percentiles, error rates)
│   └── Custom business metrics
├── AWS CloudWatch
│   ├── EC2 instance metrics
│   ├── RDS database metrics
│   ├── Lambda execution metrics
│   └── Load balancer metrics
├── Azure Monitor
│   ├── Virtual machine metrics
│   ├── App Service metrics
│   └── SQL Database metrics
├── GCP Cloud Monitoring
│   ├── Compute Engine metrics
│   ├── Cloud SQL metrics
│   └── Cloud Run metrics
├── System Logs
│   ├── Application logs (JSON, plaintext, structured)
│   ├── System logs (syslog, journalctl)
│   ├── Container logs (Docker, Kubernetes)
│   └── Database logs (PostgreSQL, MySQL, MongoDB)
├── Configuration Management Database (CMDB)
│   ├── Service dependencies
│   ├── Version control system (Git)
│   ├── Change log (deployment events)
│   └── Infrastructure topology
└── Distributed Tracing (Optional)
    ├── Jaeger traces
    ├── Zipkin traces
    └── OpenTelemetry traces
```

**Technical Specifications**:

```python
# Data Ingestion Interface
class DataIngestionManager:
    """
    Capabilities:
    - Real-time metric streaming at 5-minute intervals
    - Batch processing of historical data (up to 90 days)
    - Support for multiple data source simultaneously
    - Automatic data format conversion and normalization
    - Rate limiting to prevent overload (max 10,000 metrics/minute)
    """
    
    def ingest_prometheus_metrics(self) -> MetricsDataFrame:
        """
        Queries Prometheus API for metrics
        
        Query Pattern:
        - Fetches last N days of data
        - Resolution: 5 minutes (300 seconds)
        - Handles metric relabeling
        - Merges multiple time series
        
        Output Format: pandas DataFrame
        Shape: (num_timesteps, num_metrics)
        Columns: timestamp, metric_1, metric_2, ..., metric_n
        """
        pass
    
    def ingest_cloudwatch_metrics(self) -> MetricsDataFrame:
        """
        Fetches from AWS CloudWatch
        
        Specifications:
        - Namespace filtering (e.g., 'AWS/EC2', 'AWS/RDS')
        - Dimensions parsing (e.g., InstanceId, DBInstanceIdentifier)
        - Statistics: Average, Maximum, Minimum, Sum
        - Period: 5 minutes (300 seconds) or 1 minute (60 seconds)
        - Up to 2 weeks of historical data
        
        Rate Limits:
        - Max 400 requests/5 seconds per API
        - Implements exponential backoff
        """
        pass
    
    def ingest_logs(self) -> pd.DataFrame:
        """
        Parse and normalize logs
        
        Input Formats:
        - JSON structured logs (recommended)
        - Plaintext with regex patterns
        - Key-value pairs
        - CSV format
        
        Processing:
        - Extract timestamps (supports 15+ formats)
        - Parse severity levels (DEBUG, INFO, WARN, ERROR)
        - Extract stack traces and error messages
        - Group into 5-minute windows
        - Generate log templates (group similar logs)
        
        Output:
        DataFrame with columns:
        - timestamp: 5-minute bucket
        - log_template: normalized log pattern
        - count: occurrences in bucket
        - error_count: number of ERROR level logs
        """
        pass
    
    def ingest_deployment_events(self) -> pd.DataFrame:
        """
        Track code deployments and config changes
        
        Sources:
        - Git commits and tags
        - Deployment pipeline events
        - Configuration management system changes
        - Feature flag toggles
        
        Enrichment:
        - Author information
        - Changed files/components
        - Risk level assessment
        - Rollback capability
        
        Output:
        DataFrame with:
        - deployment_time: when deployed
        - component: service/module affected
        - version: code version
        - risk_level: HIGH, MEDIUM, LOW
        """
        pass
    
    def ingest_system_topology(self) -> Dict:
        """
        Load service dependency graph
        
        Information:
        - Service-to-service dependencies
        - Database connections
        - Cache layer relationships
        - Message queue producers/consumers
        - Load balancer targets
        
        Format: JSON graph structure
        Example:
        {
            "services": {
                "api-server": {
                    "dependencies": ["db", "cache"],
                    "type": "stateless",
                    "criticality": "high"
                }
            },
            "edges": [
                {"source": "api-server", "target": "db", "latency_sla": "100ms"}
            ]
        }
        """
        pass
    
    def validate_data_quality(self) -> DataQualityReport:
        """
        Check data completeness and consistency
        
        Checks:
        - Missing values (% of data points)
        - Duplicate timestamps
        - Out-of-order timestamps
        - Outliers in metric ranges
        - Sudden metric disappearances
        
        Report:
        - Issues found with severity levels
        - Automatic remediation recommendations
        - Data quality score (0-100%)
        """
        pass
    
    def normalize_data(self) -> NormalizedDataset:
        """
        Standardize data format and ranges
        
        Operations:
        - Convert all timestamps to UTC
        - Fill missing values (forward fill for <5min gaps)
        - Interpolate for medium gaps (5-60min)
        - Normalize metric values to [0, 1] using min-max scaling
        - Remove outliers (>5 standard deviations)
        
        Output: Standardized, ready-for-modeling dataset
        """
        pass
```

**Functional Requirements**:
- [ ] System must ingest metrics from Prometheus every 5 minutes
- [ ] System must ingest metrics from CloudWatch with <10 minute latency
- [ ] System must support 100+ metrics simultaneously
- [ ] System must handle metric name changes gracefully
- [ ] System must validate data completeness before processing
- [ ] System must detect and flag suspicious data patterns

---

#### **1.1.2 Anomaly Detection Engine**

**Requirement**: Identify metrics deviating from normal operational patterns

**Scope**:

```
ANOMALY TYPES DETECTABLE:

A. CPU & Compute Anomalies (10 types)
   1. CPU Saturation: Sustained CPU >90% beyond normal patterns
      - Detection: Compare against time-of-day baseline
      - Threshold: >90% for >5 consecutive minutes
      
   2. CPU Oscillation: Rapid fluctuation between high/low
      - Detection: Variance spike in CPU readings
      - Pattern: >10 spikes per hour
      
   3. Single-Core Bottleneck: One core at 100%, others idle
      - Detection: Per-core CPU analysis
      - Pattern: One core > 99% while others < 50%
      
   4. CPU Steal Time Spike: High steal time in virtualized env
      - Detection: Monitor /proc/stat steal time
      - Threshold: >5% sustained
      
   5. Context Switch Storm: Excessive OS context switches
      - Detection: /proc/stat context_switches metric
      - Threshold: >100k switches/sec
      
   6. CPU Affinity Misconfig: Process pinned to wrong cores
      - Detection: Taskset affinity mismatch
      - Pattern: Low utilization on pinned cores
      
   7. Thermal Throttling: CPU frequency reduction due to heat
      - Detection: /proc/cpuinfo current_mhz < rated_mhz
      - Threshold: >10% frequency reduction
      
   8. Nice Value Anomaly: Critical process starved by low-priority
      - Detection: Process scheduling priority reversal
      - Pattern: High-priority task latency > baseline
      
   9. Interrupt Storm: Excessive hardware interrupts
      - Detection: /proc/interrupts analysis
      - Threshold: >50k interrupts/sec
      
   10. CPU Cache Miss Rate: L2/L3 cache misses increase
       - Detection: perf counters for cache-misses
       - Threshold: >30% miss rate increase from baseline

B. Memory Anomalies (10 types)
   1. Memory Leak: Gradual monotonic increase without release
      - Detection: Linear regression on memory usage
      - Pattern: R² > 0.95, slope > 0.5%/hour
      
   2. Out of Memory (OOM): System kills processes
      - Detection: Direct observation via /proc/meminfo
      - Threshold: Available memory < 5% of total
      
   3. Swap Thrashing: Excessive paging to disk
      - Detection: si/so (swap in/out) metrics
      - Threshold: >100 MB/min sustained
      
   4. Page Fault Storm: High rate of page faults
      - Detection: /proc/stat page_faults
      - Threshold: >100k faults/sec
      
   5. Memory Fragmentation: Available memory but allocation fails
      - Detection: Allocation failure rate
      - Pattern: Free memory > 20%, allocation failures > 0%
      
   6. Shared Memory Leak: /dev/shm exhausted
      - Detection: du -sh /dev/shm
      - Threshold: >80% of tmpfs capacity
      
   7. Buffer/Cache Explosion: Kernel caches consuming excessive memory
      - Detection: Buffers + Cached in /proc/meminfo
      - Threshold: >60% of available memory
      
   8. Huge Pages Misconfig: THP causing latency spikes
      - Detection: Transparent Huge Pages defragmentation
      - Pattern: Latency spikes every N minutes
      
   9. Memory Bandwidth Saturation: Memory bus saturated
      - Detection: Memory bandwidth utilization counters
      - Threshold: >90% bandwidth utilization
      
   10. NUMA Node Imbalance: Uneven memory distribution
        - Detection: Per-NUMA node memory analysis
        - Pattern: Node imbalance > 20%

C. Storage & I/O Anomalies (10 types)
   1. Disk Space Exhaustion: Storage >95% full
      - Detection: df -h analysis
      - Threshold: >95% used
      
   2. I/O Wait Spike: Processes blocked waiting for disk
      - Detection: iowait metric from /proc/stat
      - Threshold: >50% of CPU time
      
   3. Disk Latency Increase: Read/write latency >10x baseline
      - Detection: iostat read_await/write_await
      - Threshold: >10x increase from baseline
      
   4. IOPS Saturation: Disk operations maxed
      - Detection: iostat r/s + w/s against disk capacity
      - Threshold: >95% of rated IOPS
      
   5. I/O Pattern Shift: Sequential to random pattern change
      - Detection: Read/write request size distribution
      - Pattern: Shift from >1MB to <4KB average
      
   6. Inode Exhaustion: No free inodes despite disk space
      - Detection: df -i analysis
      - Threshold: >95% inodes used
      
   7. Write Amplification: SSD writes >> app writes
      - Detection: Ratio of physical to logical writes
      - Threshold: >5x write amplification
      
   8. Read-Ahead Misconfig: Inefficient prefetching
      - Detection: blockdev --getra vs actual access patterns
      - Pattern: Prefetched data unused
      
   9. RAID Degradation: Array running degraded
      - Detection: mdadm status checks
      - Pattern: Member count < expected
      
   10. Filesystem Corruption: Journal errors
        - Detection: dmesg/kernel logs for filesystem errors
        - Pattern: Error count increasing

D. Network Anomalies (10 types)
   1. Packet Loss: Packets dropped >1%
      - Detection: netstat -i RX-DRP/TX-DRP
      - Threshold: >1% packet loss rate
      
   2. Bandwidth Saturation: Network interface at capacity
      - Detection: ifstat or ethtool statistics
      - Threshold: >95% of interface speed
      
   3. TCP Retransmissions: High retransmission rate
      - Detection: ss -i TCP retrans metric
      - Threshold: >1% of packets retransmitted
      
   4. Connection Reset Storm: TCP connections abruptly closed
      - Detection: netstat RST flag count
      - Threshold: >10 resets/sec
      
   5. DNS Resolution Failure: DNS lookups slow/failing
      - Detection: DNS query latency and failure rate
      - Threshold: >10% failure rate OR >1000ms latency
      
   6. NAT Table Exhaustion: NAT table full
      - Detection: conntrack table utilization
      - Threshold: >90% of conntrack limit
      
   7. MTU Mismatch: Packet fragmentation
      - Detection: Path MTU discovery failures
      - Pattern: Fragmentation rate > baseline + 100%
      
   8. ARP Cache Overflow: ARP table exceeds capacity
      - Detection: ARP entry count vs system limit
      - Threshold: >90% of ARP table capacity
      
   9. Interface Errors: CRC errors, frame errors
      - Detection: ethtool -S interface errors
      - Threshold: >0 errors (any count is anomalous)
      
   10. Asymmetric Routing: Packets take different paths
        - Detection: traceroute path analysis
        - Pattern: Different paths for request/response

E. Application Layer Anomalies (10 types)
   1. Thread Pool Exhaustion: All worker threads busy
      - Detection: Active threads == pool size
      - Threshold: 100% utilization for >1 minute
      
   2. Connection Pool Saturation: DB connections maxed
      - Detection: Active connections / pool size
      - Threshold: >95% utilization
      
   3. Garbage Collection Pause: Long GC pauses
      - Detection: GC metrics from application
      - Threshold: >500ms pause time
      
   4. Event Loop Blocking: Async loop blocked
      - Detection: Event loop latency metric
      - Threshold: >100ms latency
      
   5. Circuit Breaker Activation: Dependency protection triggered
      - Detection: Circuit breaker state changes
      - Pattern: OPEN state for >5 seconds
      
   6. Rate Limiter Throttling: Requests rejected
      - Detection: HTTP 429 response code rate
      - Threshold: >1% of requests throttled
      
   7. Cache Stampede: Simultaneous cache misses
      - Detection: Cache hit ratio drop + request spike
      - Pattern: Hit ratio <50% while requests +200%
      
   8. Deadlock: Threads waiting indefinitely
      - Detection: Thread dump analysis or timeout metrics
      - Pattern: Same threads in wait state for >60 seconds
      
   9. Semaphore Contention: Shared resource bottleneck
      - Detection: Semaphore wait time metric
      - Threshold: >100ms average wait time
      
   10. Message Queue Backlog: Messages accumulating
        - Detection: Queue depth metric
        - Threshold: Queue depth increasing for >5 minutes

F. Database Anomalies (5 types - implicit in above)
   1. Query Latency Increase: Slow query detection
   2. Connection Pool Issues: Already covered
   3. Lock Contention: Slow lock acquisitions
   4. Replication Lag: Master-slave synchronization delays
   5. Transaction Timeout: Long-running transactions

G. Business Logic Anomalies (3 types)
   1. Error Rate Spike: Unexpected increase in failures
   2. Revenue Anomaly: Transactions below expected
   3. User Session Anomalies: Session count deviations
```

**Technical Specifications**:

```python
class AnomalyDetectionEngine:
    """
    Detects 50+ anomaly types using multiple detection methods
    """
    
    def __init__(self):
        self.lstm_model = LSTMAutoencoder(...)  # Trained on normal data
        self.baselines = {}  # Historical normal values
        self.thresholds = {}  # Dynamic thresholds per metric
    
    def detect_anomalies_lstm(
        self,
        current_metrics: np.ndarray,
        sequence_length: int = 60,
        anomaly_threshold: float = 0.7
    ) -> Dict[str, float]:
        """
        LSTM-based detection using reconstruction error
        
        Process:
        1. Create sequence of last 60 data points
        2. Pass through trained LSTM autoencoder
        3. Calculate reconstruction error
        4. Compare against threshold
        
        Output: {metric_name: anomaly_score (0.0-1.0)}
        Score > 0.7 = anomalous
        Score < 0.3 = normal
        Score 0.3-0.7 = uncertain (requires ensemble)
        
        Latency: <100ms per detection
        """
        pass
    
    def detect_anomalies_statistical(
        self,
        current_value: float,
        metric_name: str,
        window_size: int = 288  # 24 hours at 5-min intervals
    ) -> Tuple[bool, float]:
        """
        Statistical detection using historical baseline
        
        Methods:
        1. Z-score: (x - mean) / std > 3.0 = anomaly
        2. IQR: x < Q1 - 1.5*IQR or x > Q3 + 1.5*IQR
        3. MAD: Median Absolute Deviation
        4. Seasonal decomposition: Actual vs trend
        
        Adaptive:
        - Baseline recalculated daily
        - Accounts for time-of-day patterns
        - Handles weekday vs weekend differences
        
        Returns: (is_anomalous, z_score)
        """
        pass
    
    def detect_anomalies_temporal(
        self,
        time_series: np.ndarray,
        window_size: int = 24
    ) -> Dict:
        """
        Detect temporal patterns indicative of problems
        
        Patterns:
        1. Linear Trend: Monotonic increase/decrease
           - Memory leak pattern
           - Detect via linear regression R² > 0.95
           
        2. Sudden Spike: Sharp increase followed by plateau
           - CPU spike pattern
           - Detect via derivative spike > 5x normal
           
        3. Oscillation: High frequency fluctuation
           - Thread thrashing pattern
           - Detect via FFT spectral analysis
           
        4. Step Change: Shift in baseline
           - Config change effect
           - Detect via change point detection (CUSUM)
           
        5. Drift: Slow shift in mean/variance
           - Concept drift
           - Detect via ADWIN algorithm
        
        Returns: {pattern_type: confidence_score}
        """
        pass
    
    def ensemble_anomaly_detection(
        self,
        current_metrics: np.ndarray,
        time_series: np.ndarray,
        metric_name: str
    ) -> float:
        """
        Combine multiple detection methods for robustness
        
        Weights:
        - LSTM score: 40%
        - Statistical score: 35%
        - Temporal score: 25%
        
        Ensemble logic:
        - If all agree: high confidence (0.8-1.0)
        - If majority agree: medium confidence (0.5-0.8)
        - If disagreement: low confidence (<0.5)
        
        Returns: final_anomaly_score (0.0-1.0)
        """
        pass
    
    def detect_correlated_anomalies(
        self,
        anomalies: Dict[str, float]
    ) -> Dict[str, List[str]]:
        """
        Identify which anomalies are likely related
        
        Example:
        - If CPU spike AND memory increase: likely same root cause
        - If API latency AND DB connections: likely cascading effect
        
        Output: {metric: [correlated_metrics]}
        
        Used later for causal analysis
        """
        pass
```

**Functional Requirements**:
- [ ] System must detect all 50+ anomaly types
- [ ] System must achieve >80% precision on synthetic failures
- [ ] System must achieve >75% recall on synthetic failures
- [ ] Anomaly detection must complete within 5 minutes
- [ ] System must support custom anomaly definitions
- [ ] System must adapt to metric changes without retraining
- [ ] System must reduce false positives through ensemble methods

---

#### **1.1.3 Causal Inference Engine**

**Requirement**: Determine causal relationships between anomalies

**Scope**:

```python
class CausalInferenceEngine:
    """
    Determines which anomalies CAUSE which other anomalies
    vs. which are merely correlated or coincidental
    """
    
    def granger_causality_analysis(
        self,
        data: pd.DataFrame,
        max_lag: int = 10,
        significance_level: float = 0.05
    ) -> Dict[Tuple[str, str], CausalityResult]:
        """
        Granger Causality Test:
        X "Granger-causes" Y if:
        1. X temporally precedes Y (X_t happens before Y_t)
        2. Past values of X improve prediction of Y
           beyond using only Y's own past values
        
        Mathematically:
        Model 1: Y_t = α₀ + Σ α_i * Y_{t-i} + ε_t
        Model 2: Y_t = β₀ + Σ β_i * Y_{t-i} + Σ γ_j * X_{t-j} + ε_t
        
        Test: Does Model 2 explain significantly more variance?
        (F-test on residuals)
        
        Process:
        1. For each metric pair (X, Y)
        2. Test lags from 1 to max_lag
        3. Find lag where Granger causality strongest (lowest p-value)
        4. If p-value < significance_level: X causes Y
        
        Output: {(cause, effect): {
            'is_causal': bool,
            'best_lag': int,  # How many timesteps between cause and effect
            'p_value': float,  # Statistical significance
            'r_squared': float,  # Variance explained
            'strength': float  # Normalized [0, 1]
        }}
        
        Limitations:
        - Requires stationary time series (handle via differencing)
        - Cannot detect non-linear relationships
        - May miss instantaneous causality (lag=0)
        - Linear assumption
        
        Advantages:
        - Well-established statistical test
        - Explicitly handles temporal precedence
        - Provides p-values for confidence
        """
        pass
    
    def transfer_entropy_analysis(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        k: int = 3  # Number of previous steps
    ) -> float:
        """
        Transfer Entropy (TE):
        Information flow from X to Y
        
        TE(X→Y) = Σ p(y_n, y_past, x_past) * log(p(y_n|y_past,x_past) / p(y_n|y_past))
        
        Interpretation:
        - Higher TE = X influences Y more
        - TE = 0 = No information transfer
        - TE > 0 = X provides information about future Y
        
        Advantages over Granger Causality:
        - Detects non-linear relationships
        - Information-theoretic foundation
        - Handles symbolic data
        
        Disadvantages:
        - Computationally expensive
        - Sensitive to binning/parameters
        - Limited theoretical guarantees
        
        Output: transfer_entropy_score (0.0 to ∞)
        """
        pass
    
    def pc_algorithm_causal_discovery(
        self,
        data: pd.DataFrame,
        significance_level: float = 0.05
    ) -> nx.Graph:
        """
        PC Algorithm (Peter-Clark):
        Learn causal structure from observational data
        
        Algorithm:
        Phase 1: Start with fully connected undirected graph
        
        Phase 2: Remove edges based on conditional independence
        For each node pair (i, j):
            For each subset S of neighbors of i (excluding j):
                If i ⊥ j | S (conditionally independent):
                    Remove edge i-j
                    Break (found separating set)
        
        Phase 3: Orient edges to form v-structures
        For unshielded triple (i,k,j) where:
        - i-k, k-j exist
        - i-j NOT adjacent
        - i doesn't cause j (not in same confounded component):
            Orient as i → k ← j (v-structure)
        
        Phase 4: Propagate orientations
        Apply Meek rules to orient remaining edges
        
        Input: Data with n metrics
        Output: Directed Acyclic Graph (DAG)
        
        Advantages:
        - Provably correct under assumptions
        - Handles non-linear relationships
        - Computes minimal sufficient statistics
        
        Disadvantages:
        - Assumes no hidden confounders
        - Sensitive to false conditional independence
        - Computational complexity: O(3^n)
        - Needs large samples
        
        Assumptions (must be verified):
        1. No hidden/latent confounders
        2. Data is i.i.d. from stationary distribution
        3. Causal sufficiency (all causes observed)
        4. No cycles (acyclic structure)
        
        Conditional Independence Test:
        Uses partial correlation:
        - H0: X ⊥ Y | Z (independent)
        - H1: X ⊥̸ Y | Z (dependent)
        
        Test via:
        - Partial correlation > threshold
        - p-value < significance_level
        - Works for Gaussian data; generalize via copulas
        """
        pass
    
    def temporal_precedence_analysis(
        self,
        anomalies: Dict[str, int],  # metric: first detection index
        causal_edges: List[Tuple[str, str]]
    ) -> Dict[Tuple[str, str], bool]:
        """
        Verify temporal precedence for each causal edge
        
        Rule: Cause must precede effect
        
        Process:
        1. For each edge (X→Y)
        2. Check: first_anomaly_time[X] < first_anomaly_time[Y]
        3. If violated: remove edge (impossible causality)
        
        Also check: time lag is reasonable
        - Medical max lag: 1-5 timesteps (5-25 minutes)
        - System-dependent: could be 1-100 steps
        
        Output: {(X, Y): is_temporally_valid}
        """
        pass
    
    def latency_estimation(
        self,
        causal_edges_with_lags: Dict[Tuple[str, str], int]
    ) -> Dict[Tuple[str, str], int]:
        """
        Estimate propagation delay for each causal edge
        
        Interpretation:
        - Database query latency → API timeout: ~5 minutes latency
        - Memory leak → GC pause: ~hours
        - Network partition → error spike: ~immediate
        
        Used for:
        1. Validating causal structure
        2. Predicting when symptoms will manifest
        3. Prioritizing investigation (fast-acting causes first)
        
        Estimation methods:
        1. Granger causality best_lag
        2. Cross-correlation peak lag
        3. Domain knowledge templates
        
        Output: {(cause, effect): estimated_lag_in_minutes}
        """
        pass
    
    def build_causal_graph(
        self,
        granger_results: Dict,
        pc_graph: nx.Graph,
        temporal_validity: Dict,
        confidence_threshold: float = 0.7
    ) -> nx.DiGraph:
        """
        Combine multiple causal inference methods
        
        Strategy: Ensemble approach
        
        For each potential edge (X, Y):
        1. Check Granger causality p-value
        2. Check PC algorithm learned structure
        3. Check temporal precedence
        4. Check domain knowledge graph (from CMDB)
        
        Voting:
        - All methods agree: high confidence (0.9+)
        - 2 of 3 agree: medium confidence (0.7-0.9)
        - 1 of 3 agree: low confidence (0.3-0.7)
        - Conflict: edge removed
        
        Edge weights: confidence scores
        
        Output: Directed Acyclic Graph (DAG)
        
        Properties:
        - Nodes: system metrics
        - Edges: causal relationships with weights
        - Path from root cause to symptoms
        """
        pass
    
    def validate_causal_assumptions(
        self,
        data: pd.DataFrame
    ) -> Dict[str, any]:
        """
        Check if causal inference assumptions are satisfied
        
        Assumptions to validate:
        1. No hidden confounders
           - Test: Are all relevant metrics included?
           - Check: Correlation of residuals
           
        2. Acyclicity
           - Test: Graph contains no cycles
           - Method: Topological sort
           
        3. No selection bias
           - Test: Data collection methodology
           - Check: No missing data patterns
           
        4. Stationarity
           - Test: Augmented Dickey-Fuller test
           - Remedy: Apply differencing
           
        5. No autocorrelation in residuals
           - Test: Ljung-Box test
           - Remedy: Include lagged variables
        
        Output: {
            'assumptions_satisfied': bool,
            'violations': [list of violations],
            'remedies': [recommended corrections],
            'confidence_adjustment': float
        }
        
        If assumptions violated, reduce confidence in results
        """
        pass
```

**Functional Requirements**:
- [ ] System must distinguish correlation from causation
- [ ] System must identify causal relationships with >70% accuracy
- [ ] System must handle temporal delays between cause and effect
- [ ] System must validate causal assumptions
- [ ] System must identify v-structures and confounders
- [ ] Causal inference must complete within 10 minutes

---

#### **1.1.4 Root Cause Ranking & Scoring**

**Requirement**: Rank candidate anomalies as potential root causes

```python
class RootCauseRanker:
    """
    Score and rank all anomalies as potential root causes
    Output: Top N root causes with confidence scores
    """
    
    def calculate_scores(
        self,
        anomalies: Dict[str, float],  # metric: anomaly_score
        causal_graph: nx.DiGraph,
        anomaly_detection_times: Dict[str, int],
        earliest_failure_time: int,
        deployment_events: pd.DataFrame = None,
        config_changes: pd.DataFrame = None
    ) -> List[RootCauseScore]:
        """
        Multi-factor scoring framework
        
        Scoring factors:
        1. Causal Outflow (40% weight)
           - How many downstream metrics does this cause?
           - More descendants = likely root cause
           - Calculation: descendants / total_metrics
           - Range: [0.0, 1.0]
           
        2. Causal Inflow (20% weight)
           - How many upstream causes does this have?
           - Fewer ancestors = likely root cause
           - Calculation: 1.0 - (ancestors / total_metrics)
           - Range: [0.0, 1.0]
           
        3. Temporal Priority (30% weight)
           - How early did this anomaly appear?
           - Earlier appearance = more likely root cause
           - Calculation: 1.0 - (detection_time - earliest_time) / max_possible_delay
           - Range: [0.0, 1.0]
           
           Example:
           - Earliest anomaly detected at T=0: temporal_score = 1.0 (perfect)
           - Anomaly at T=100: temporal_score = 0.9 (if max_delay=1000)
           
        4. Anomaly Severity (5% weight)
           - How far from normal is this metric?
           - Higher deviation = stronger signal
           - Calculation: Direct from anomaly detection score
           - Range: [0.0, 1.0]
           
        5. Event Correlation (5% weight)
           - Did deployment/config change precede this anomaly?
           - Recent change = higher probability of causation
           - Calculation: 1.0 / (1.0 + time_since_event_minutes)
           - Range: [0.0, 1.0]
        
        Final Score = 0.40*outflow + 0.20*inflow + 0.30*temporal + 0.05*severity + 0.05*event
        
        Output: List sorted by score (descending)
        [{
            'metric': 'db_query_latency',
            'rank': 1,
            'final_score': 0.87,
            'scores': {
                'outflow': 0.85,
                'inflow': 0.90,
                'temporal': 1.0,
                'severity': 0.91,
                'event': 0.0
            },
            'confidence': 'High',
            'downstream_effects': ['db_connections', 'api_latency', ...],
            'justification': "..."
        }]
        """
        pass
    
    def calculate_confidence_level(
        self,
        final_score: float,
        agreement_level: float,  # % of detection methods agreeing
        causal_chain_quality: float,
        temporal_validity: bool
    ) -> str:
        """
        Determine overall confidence level
        
        Levels:
        - "Critical" (95-100%): All signals aligned, clear root cause
        - "High" (85-95%): Most signals agree, strong evidence
        - "Medium" (70-85%): Partial signals, probable root cause
        - "Low" (50-70%): Weak signals, possible root cause
        - "Very Low" (<50%): Conflicting signals, uncertain
        
        Calculation:
        confidence = avg(final_score, agreement_level, causal_chain_quality)
        
        Threshold adjustments:
        - If temporal_validity = False: reduce by 10%
        - If deployment within 1 hour: increase by 5%
        """
        pass
    
    def generate_justification(
        self,
        root_cause: str,
        scores: Dict,
        causal_chain: List[str],
        supporting_evidence: Dict
    ) -> str:
        """
        Generate human-readable explanation for root cause ranking
        
        Format:
        
        "We identified {root_cause} as the primary root cause with
        {confidence}% confidence for the following reasons:
        
        1. TEMPORAL PRECEDENCE
           - This metric first deviated from normal at {time}
           - Other anomalies appeared {delay} minutes later
        
        2. CAUSAL IMPACT
           - This anomaly directly caused {N} downstream failures:
             * {effect1}: {description}
             * {effect2}: {description}
           - Causal chain: {root_cause} → {step1} → {step2} → {symptom}
        
        3. STATISTICAL EVIDENCE
           - Anomaly detected with {detection_confidence}% confidence
           - Granger causality test p-value: {p_value}
           - {N} independent detection methods agree
        
        4. CONTEXTUAL EVIDENCE
           - Deployment {version} happened {time_ago} minutes ago
           - Files changed: {list_of_files}
           - Config change detected at {time}
        
        5. SUPPORTING METRICS
           - Baseline: {baseline_value}
           - Observed: {observed_value}
           - Deviation: {deviation_pct}%
        
        RECOMMENDED ACTIONS
        1. Immediate: {immediate_action}
        2. Short-term: {short_term_action}
        3. Long-term: {long_term_action}"
        """
        pass
```

**Functional Requirements**:
- [ ] System must rank root causes with >70% top-1 accuracy
- [ ] System must provide >85% top-3 accuracy
- [ ] Ranking must complete within 5 minutes
- [ ] System must provide confidence scores for each root cause
- [ ] System must explain ranking with supporting evidence

---

#### **1.1.5 Causal Chain Tracing**

**Requirement**: Map complete failure propagation path

```python
class CausalChainTracer:
    """
    Trace path from root cause through all intermediate steps
    to final user-visible symptoms
    """
    
    def trace_failure_propagation(
        self,
        root_cause: str,
        causal_graph: nx.DiGraph
    ) -> List[CausalPath]:
        """
        Find all paths from root cause to leaf nodes (symptoms)
        
        Algorithm: DFS to find all paths
        
        Example output:
        [
            {
                'path': ['db_query_latency', 'db_connections', 'api_latency', 'error_rate'],
                'steps': [
                    {
                        'from': 'db_query_latency',
                        'to': 'db_connections',
                        'lag': 5,  # minutes
                        'mechanism': 'Slow queries hold connections longer'
                    },
                    {
                        'from': 'db_connections',
                        'to': 'api_latency',
                        'lag': 10,
                        'mechanism': 'Connection pool exhaustion forces queuing'
                    },
                    ...
                ]
            },
            ...
        ]
        
        Usage:
        1. Show engineers the complete failure sequence
        2. Estimate time to symptom manifestation
        3. Identify intervention points
        """
        pass
    
    def estimate_symptom_time(
        self,
        causal_path: List[str],
        lag_estimates: Dict[Tuple[str, str], int],
        root_cause_time: int
    ) -> Dict[str, int]:
        """
        Estimate when each step in causal chain will manifest
        
        Calculation:
        For each step in path:
            step_time = root_cause_time + sum(lags up to this step)
        
        Example:
        - Root cause at T=0 (db query latency)
        - +5 min lag → T=5 (connection pool exhaustion)
        - +10 min lag → T=15 (API latency)
        - +5 min lag → T=20 (error rate spike)
        
        Output: {metric: estimated_detection_time}
        
        Usage:
        1. Predict when symptoms will manifest
        2. Estimate time available for remediation
        3. Prioritize engineer actions
        """
        pass
    
    def identify_intervention_points(
        self,
        causal_path: List[str],
        system_capabilities: Dict
    ) -> List[InterventionPoint]:
        """
        Identify where engineers can intervene to stop propagation
        
        Intervention types:
        1. Prevent root cause
           - Cost: High (requires root fix)
           - Time: Long (requires development/deployment)
           - Effectiveness: 100% (prevents all downstream)
           - Example: "Fix the slow query with index"
        
        2. Break intermediate link
           - Cost: Medium (requires immediate action)
           - Time: Medium (minutes to hours)
           - Effectiveness: Partial (stops downstream cascade)
           - Example: "Circuit breaker to fail fast instead of timeout"
        
        3. Prevent symptom manifestation
           - Cost: Low (quick mitigations)
           - Time: Very short (immediate)
           - Effectiveness: Low (users still affected, hidden)
           - Example: "Increase error rate threshold to delay alert"
        
        Output: [{
            'step': 'db_connections',
            'type': 'Break intermediate link',
            'action': 'Circuit breaker to fail immediately',
            'cost': 'Low',
            'time_to_implement': '5 minutes',
            'effectiveness': 'High',
            'side_effects': ['Some requests will return errors immediately']
        }]
        """
        pass
```

**Functional Requirements**:
- [ ] System must identify complete causal chains
- [ ] System must estimate propagation delays between steps
- [ ] System must identify multiple failure paths
- [ ] System must suggest intervention points

---

#### **1.1.6 Report Generation & Visualization**

**Requirement**: Present findings in actionable formats

```python
class ReportGenerator:
    """
    Generate reports in multiple formats for different audiences
    """
    
    def generate_executive_summary(
        self,
        root_causes: List[RootCauseScore],
        failure_duration: int,
        estimated_impact_dollars: float
    ) -> str:
        """
        1-page summary for executive leadership
        
        Format:
        - What happened: Brief description
        - When: Timestamp and duration
        - Cost: Estimated business impact
        - Root cause: Top 1-2 causes
        - Status: Resolved/In-progress/Investigation
        - Action: What's being done
        """
        pass
    
    def generate_technical_report(
        self,
        root_causes: List[RootCauseScore],
        causal_chains: List[CausalPath],
        anomalies: Dict,
        metrics_timeline: pd.DataFrame,
        supporting_evidence: Dict
    ) -> str:
        """
        Detailed technical report for engineers
        
        Sections:
        1. Incident Timeline
           - When each anomaly detected
           - Magnitude of deviations
           
        2. Root Cause Analysis
           - Primary root cause with confidence
           - Contributing factors
           - Evidence for each
           
        3. Failure Propagation
           - Complete causal chains
           - Timing of each step
           - Mechanisms connecting steps
           
        4. Anomaly Details
           - Metrics that deviated
           - Baseline vs actual values
           - Anomaly scores from multiple methods
           
        5. Recommendations
           - Immediate remediation
           - Short-term fixes
           - Long-term prevention
           
        6. Appendix
           - Supporting graphs
           - Full timeline
           - All metrics examined
        """
        pass
    
    def generate_json_report(self) -> Dict:
        """
        Structured JSON for integration with tools
        
        Schema:
        {
            'incident_id': 'INC-2024-001',
            'timestamp': '2024-02-23T15:30:00Z',
            'duration_minutes': 45,
            'status': 'resolved',
            'root_causes': [
                {
                    'rank': 1,
                    'metric': 'db_query_latency',
                    'confidence': 0.87,
                    'p_value': 0.001,
                    'evidence': [...]
                }
            ],
            'causal_chains': [...],
            'metrics_timeline': [...],
            'recommendations': [...]
        }
        """
        pass
    
    def generate_interactive_dashboard(
        self,
        metrics_data: pd.DataFrame,
        anomaly_scores: Dict[str, List[float]],
        causal_graph: nx.DiGraph,
        root_causes: List[RootCauseScore]
    ) -> StreamlitApp:
        """
        Interactive web dashboard for investigation
        
        Features:
        1. Timeline slider
           - Scrub through failure timeline
           - See metrics at each point in time
           - View anomaly detection results
        
        2. Metric explorer
           - Select metrics to examine
           - View timeseries with anomalies highlighted
           - Compare to historical baseline
        
        3. Causal graph visualization
           - Interactive graph of causal relationships
           - Highlight root cause in red
           - Show propagation paths
           - Hover for details
        
        4. Root cause details
           - Detailed justification
           - Supporting evidence
           - Alternative hypotheses
           - Confidence score explanation
        
        5. Recommendations
           - Immediate actions
           - Who should do what
           - Estimated time to fix
        """
        pass
```

**Functional Requirements**:
- [ ] System must generate executive summary within 5 minutes
- [ ] System must generate technical report within 10 minutes
- [ ] Reports must be in PDF, JSON, and HTML formats
- [ ] System must provide interactive dashboard
- [ ] Reports must include confidence intervals and caveats

---

### 1.2 Limitations (What the System CANNOT Do)

#### **1.2.1 Inherent Limitations**

**The system is NOT designed to**:

| Limitation | Why | Mitigation |
|-----------|-----|-----------|
| **Identify Hidden Confounders** | Causal inference assumes all relevant metrics are included. If unmeasured variable causes both metrics, algorithm can't detect it. Example: Coordinator failure causes both API latency AND db latency without direct connection. | Include comprehensive metrics coverage; use domain knowledge to identify likely confounders |
| **Handle Black-box Systems** | If internal workings unknown, causal paths can't be traced. Example: Third-party SaaS with only error codes visible. | Integrate service-level APIs when available; maintain detailed documentation of system behavior |
| **Distinguish Simultaneous Causes** | If multiple independent root causes trigger simultaneously (T=0 for both), system can't determine which is "primary." | Flag as "Multiple Root Causes"; recommend parallel investigation |
| **Predict Failures** | System is reactive, not predictive. It analyzes existing failures, not future risks. | Implement separate predictive monitoring system; use RCA insights to improve predictions |
| **Detect Logic Errors** | If code logic is wrong but all metrics look normal, RCA can't help. Example: "SELECT * FROM users WHERE age > 0" returns children. | Integrate application-level assertions; use synthetic transaction monitoring |
| **Handle Non-Linear Relationships** | Statistical tests assume linear causality. Non-linear relationships may be missed. Example: "Latency increases exponentially with DB load above threshold." | Implement neural network-based causal discovery for complex cases; use domain knowledge |
| **Determine Fault Severity** | System identifies causes, not impact magnitude. A metric spike might be harmless or catastrophic depending on context. | Correlate with business metrics; integrate SLO/SLA thresholds |
| **Understand Business Impact** | System sees technical metrics, not revenue/customers lost. "Error rate 10%" might mean $1 or $1M depending on what errors. | Integrate business metric tracking; map technical failures to customer experience |
| **Cross-Organization Failures** | If failure involves multiple organizations (e.g., vendor API failure), system only sees local impact. | Integrate external incident feeds; maintain vendor status pages; implement synthetic tests |
| **Detect Subtle Configuration Issues** | If failure caused by subtle misconfiguration that doesn't obviously deviate from normal, may be missed. Example: "Connection pool size set to 95 instead of 100, works until traffic spike." | Implement configuration drift detection; version control all configs; test edge cases |

---

#### **1.2.2 Data Limitations**

**System requires**:
- Minimum 30 days of normal operational data for training baseline
- Consistent metric collection (5-minute intervals)
- <5% missing data points
- Sufficient volume of failure events for evaluation

**System performance degrades with**:
- New services with <30 days history
- Sparse metrics (few data points)
- Inconsistent metric naming/availability
- High seasonality without pattern modeling
- Multiple failure modes overlapping

---

#### **1.2.3 Temporal Limitations**

| Scenario | Capability | Limitation |
|----------|-----------|-----------|
| **Slow onset (days)** | Detects via LSTM trend detection | May take long time to identify cause |
| **Fast cascades (seconds)** | Limited by 5-minute collection interval; may miss intermediate steps | Requires sub-minute metrics for full trace |
| **Distributed latencies (hours)** | Can handle via lag estimation | Requires complete metric set over full duration |
| **Instantaneous failures** | Cannot distinguish between simultaneous causes | Flag as "simultaneous root causes" |

---

#### **1.2.4 Methodological Limitations**

**Causal Inference Assumptions** (if violated, confidence drops):
1. **Acyclicity**: System assumes no feedback loops
   - Violation: Microservices A → B → C → A (retries)
   - Impact: Cannot determine true root cause direction
   - Mitigation: Detect and remove cycle edges; use domain knowledge for direction

2. **No Hidden Confounders**: All causal factors are measured
   - Violation: GC pause caused by external process using CPU
   - Impact: May misidentify gc_pause as primary cause instead of external process
   - Mitigation: Ensure comprehensive metric coverage

3. **Stationarity**: Statistical properties don't change over time
   - Violation: Traffic pattern changes (e.g., new feature launch)
   - Impact: Baseline may be invalid; false anomalies
   - Mitigation: Retrain models regularly; detect concept drift

4. **No Selection Bias**: Failures are representative of true failure distribution
   - Violation: Only sampled 1% of requests
   - Impact: Biased conclusions
   - Mitigation: Ensure complete data collection

5. **Linearity**: Relationships between metrics are approximately linear
   - Violation: API latency = 10ms baseline + 0.001 * CPU²
   - Impact: Statistical tests may fail
   - Mitigation: Transform variables; use non-linear detection methods

---

#### **1.2.5 System-Level Limitations**

**What RCA cannot do**:
- [ ] Fix the problem (requires human intervention)
- [ ] Guarantee correct diagnosis (requires validation)
- [ ] Replace domain expertise (requires expert confirmation)
- [ ] Handle real-time requirements (<1 minute analysis)
- [ ] Scale to >1000 metrics without degradation
- [ ] Work on completely new systems with no baseline data

---

### 1.3 Out of Scope

**The following are explicitly OUT OF SCOPE for v1.0**:

1. **Automated Remediation**
   - System identifies causes, does NOT automatically fix
   - Example: Detects memory leak but doesn't restart service
   - Rationale: Too risky without human approval; requires domain knowledge

2. **Predictive Analytics**
   - System is reactive to failures, not proactive
   - Does not predict future failures
   - Rationale: Different problem space; separate product

3. **Multi-Tenant Isolation**
   - RCA analysis may reveal information across tenants
   - Not suitable for multi-tenant SaaS without data segregation
   - Rationale: Security implications; handled in v2.0

4. **Real-Time Streaming Analysis**
   - System operates on batches (5-minute intervals)
   - Not suitable for sub-second anomalies
   - Rationale: Statistical methods require sufficient samples

5. **Unsupervised Anomaly Detection for New Metrics**
   - New metrics without baseline data can't be analyzed
   - Requires 30+ days of normal data first
   - Rationale: Insufficient information for pattern learning

6. **Business Logic Understanding**
   - System doesn't understand application logic
   - Can't distinguish "error rate 10% = bad" vs "error rate 10% = normal for this operation"
   - Rationale: Requires manual SLO/SLA definition

7. **Multi-Failure Scenarios**
   - If multiple independent failures occur simultaneously, system defaults to "unclear"
   - Can't reliably distinguish cause when >3 independent anomalies present
   - Rationale: Combinatorial explosion; requires different algorithm

8. **Custom Metrics Integration**
   - System works with standard monitoring platforms
   - Custom application metrics require additional setup
   - Rationale: Data model assumptions

---

## 2. SECURITY & SAFETY REQUIREMENTS

### 2.1 Overview

The RCA system has access to sensitive production data and could cause damage if:
1. **Exploited**: Used to trigger false alerts or hide real failures
2. **Malfunctioning**: Generates incorrect diagnoses leading to wrong remediation
3. **Overwhelmed**: Resource exhaustion prevents failure detection
4. **Misdirected**: Provides recommendations to wrong team/escalation

**Defense Principle**: "Defense in Depth"
- Multiple layers of protection
- Assume individual components may fail
- Graceful degradation

---

### 2.2 Data Security

#### **2.2.1 Access Control**

**Requirement**: Only authorized personnel can view RCA results

**Implementation**:

```python
class AccessControl:
    """
    Role-based access control for RCA system
    """
    
    ROLES = {
        'viewer': {
            'can_view_reports': True,
            'can_view_metrics': True,
            'can_export_reports': False,
            'can_modify_settings': False,
            'can_delete_data': False
        },
        'analyst': {
            'can_view_reports': True,
            'can_view_metrics': True,
            'can_export_reports': True,
            'can_modify_settings': True,
            'can_delete_data': False
        },
        'sre': {
            'can_view_reports': True,
            'can_view_metrics': True,
            'can_export_reports': True,
            'can_modify_settings': True,
            'can_delete_data': False,
            'can_trigger_remediation': True  # Future feature
        },
        'admin': {
            'can_do_anything': True
        }
    }
    
    def check_permission(
        self,
        user_id: str,
        user_role: str,
        action: str
    ) -> bool:
        """
        Verify user has permission for action
        
        Process:
        1. Get user's role
        2. Check if role permits action
        3. Log attempt (for audit)
        4. Return true/false
        """
        if user_role not in self.ROLES:
            return False
        
        permissions = self.ROLES[user_role]
        has_permission = permissions.get(action, False)
        
        # Log attempt
        self.audit_log(user_id, action, has_permission)
        
        return has_permission
    
    def audit_log(self, user_id: str, action: str, granted: bool):
        """
        Log all access attempts for security audit
        
        Logged info:
        - Who (user_id)
        - What (action)
        - When (timestamp)
        - Result (granted/denied)
        - IP address
        - User agent
        
        Retention: 90 days minimum
        """
        pass
    
    def enforce_data_segregation(
        self,
        user_id: str,
        team: str
    ) -> List[str]:
        """
        Ensure users only see RCA data for their team's services
        
        Rules:
        1. Analysts can only see failures in services their team owns
        2. Cross-team escalations must be approved by manager
        3. Admin can see all (with explicit logging)
        
        Example:
        - Backend team analyst: can see api-server, auth-service
        - Database team analyst: can see db-cluster, replication-service
        - Infrastructure team: can see all (owns infrastructure)
        
        Returns: List of services user can analyze
        """
        pass
```

**Requirements**:
- [ ] All access to RCA dashboard requires authentication
- [ ] User roles enforced: viewer, analyst, SRE, admin
- [ ] Data segregated by team/service ownership
- [ ] Cross-team escalations logged and approved
- [ ] All access attempts logged for audit
- [ ] Audit logs retained for 90+ days

---

#### **2.2.2 Data Encryption**

**Requirement**: Sensitive metrics and logs encrypted in transit and at rest

**Implementation**:

```python
class DataEncryption:
    """
    Encrypt sensitive production data
    """
    
    def encrypt_at_rest(
        self,
        data: pd.DataFrame,
        encryption_key: str
    ) -> bytes:
        """
        Encrypt data stored in database/filesystem
        
        Algorithm: AES-256-GCM (authenticated encryption)
        - Provides confidentiality (can't read without key)
        - Provides authenticity (detects tampering)
        - Provides integrity (detects corruption)
        
        Key management:
        - Keys stored in Key Management Service (AWS KMS, GCP Cloud KMS)
        - Never stored in code or configuration
        - Rotated every 90 days
        - Access to master key logged and restricted
        
        Process:
        1. Serialize data to JSON
        2. Compress (usually reduces by 50%)
        3. Encrypt with AES-256-GCM
        4. Store encrypted ciphertext + authentication tag
        """
        # Pseudocode
        serialized = json.dumps(data)
        compressed = zlib.compress(serialized)
        
        cipher = AES.new(encryption_key, AES.MODE_GCM)
        nonce = cipher.nonce  # Random initialization vector
        
        ciphertext, tag = cipher.encrypt_and_digest(compressed)
        
        # Return: nonce + ciphertext + tag (needed for decryption)
        return nonce + ciphertext + tag
    
    def encrypt_in_transit(
        self,
        data: any,
        recipient_certificate: str
    ) -> bytes:
        """
        Encrypt data sent over network
        
        Protocol: TLS 1.3 (minimum)
        - Automatically encrypts all HTTP traffic
        - Certificate pinning for API calls
        - Perfect forward secrecy (old sessions unbreakable)
        
        Certificate requirements:
        - Valid certificate for domain
        - Signed by trusted CA
        - Not expired
        - Certificate pinning for critical APIs
        
        Process:
        1. All HTTP traffic over HTTPS only
        2. HTTP requests redirected to HTTPS
        3. HSTS header set (force HTTPS for 1 year)
        4. Certificate pinned for internal APIs
        """
        pass
    
    def encrypt_logs(
        self,
        log_message: str,
        sensitive_fields: List[str]
    ) -> str:
        """
        Redact sensitive data from logs before encryption
        
        Sensitive fields:
        - Customer names/IDs
        - API keys/tokens
        - Database connection strings
        - Credit card numbers
        - Personal information (PII)
        
        Process:
        1. Identify sensitive fields
        2. Replace with [REDACTED] or hash
        3. Log redacted version
        4. Keep original in encrypted vault (if needed for investigation)
        
        Example:
        Before: "Database error: cannot connect to user=admin pass=secret123 host=db.example.com"
        After: "Database error: cannot connect to user=[REDACTED] pass=[REDACTED] host=[REDACTED]"
        
        Vault entry: {
            'incident_id': 'INC-001',
            'timestamp': '2024-02-23T15:30:00Z',
            'encrypted_data': <encrypted connection string>,
            'key_used': 'kms-key-v1',
            'accessed_by': ['security-team-lead'],
            'reason': 'Investigation of failed authentication'
        }
        """
        pass
    
    def key_rotation(
        self,
        old_key_id: str,
        new_key_id: str
    ):
        """
        Rotate encryption keys without downtime
        
        Process (Zero-Downtime Rotation):
        1. Generate new key in KMS
        2. Mark old key as "deprecated"
        3. Continue using old key for reading/decryption
        4. New data encrypted with new key
        5. Background job re-encrypts old data with new key
        6. After all data re-encrypted, old key can be deleted
        
        Frequency: Every 90 days
        """
        pass
```

**Requirements**:
- [ ] All metrics at rest encrypted with AES-256-GCM
- [ ] All metrics in transit encrypted with TLS 1.3
- [ ] Encryption keys stored in KMS, never in code
- [ ] Keys rotated every 90 days
- [ ] Sensitive data redacted from logs
- [ ] Certificate pinning for critical APIs

---

#### **2.2.3 Data Retention & Purging**

**Requirement**: Clear data retention policy and automatic purging

**Implementation**:

```python
class DataRetention:
    """
    Manage data retention lifecycle
    """
    
    RETENTION_POLICY = {
        'raw_metrics': {
            'duration_days': 30,
            'reason': 'Insufficient disk for longer retention; aggregated data kept longer',
            'storage': 'hot_storage'
        },
        'aggregated_metrics': {
            'duration_days': 365,
            'reason': 'Daily aggregates for trend analysis over 1 year',
            'storage': 'warm_storage'
        },
        'rca_reports': {
            'duration_days': 365,
            'reason': 'Keep for incident investigation and pattern analysis',
            'storage': 'warm_storage'
        },
        'audit_logs': {
            'duration_days': 2555,  # 7 years
            'reason': 'Legal compliance for financial systems',
            'storage': 'cold_storage'
        },
        'deployment_events': {
            'duration_days': 365,
            'reason': 'Historical deployments for correlation with failures',
            'storage': 'warm_storage'
        },
        'user_access_logs': {
            'duration_days': 365,
            'reason': 'Audit trail for access control violations',
            'storage': 'warm_storage'
        }
    }
    
    def purge_old_data(self):
        """
        Delete data older than retention period
        
        Process:
        1. Identify data past retention date
        2. Before deletion:
           a. Create backup (in case accidental deletion)
           b. Log deletion event
           c. Verify data archival if required
        3. Permanently delete (secure deletion, overwrite with random data)
        4. Verify deletion (attempt to read deleted records)
        5. Log completion
        
        Frequency: Daily at 2 AM UTC
        """
        pass
    
    def archive_data(
        self,
        data: pd.DataFrame,
        archive_type: str  # 'warm', 'cold'
    ) -> str:
        """
        Archive old data to cheaper storage
        
        Storage tiers:
        - Hot: Fast SSD storage, expensive, for <30 days data
        - Warm: Standard storage, medium cost, for <365 days data
        - Cold: Archival storage (AWS Glacier, GCP Coldline), cheap, for >365 days
        
        Process:
        1. Compress data (usually 50-90% reduction)
        2. Encrypt
        3. Upload to archival storage
        4. Delete from hot storage
        5. Track archive location in catalog
        
        Retrieval:
        - Warm storage: Immediate
        - Cold storage: 4-24 hour delay (acceptable for legal holds)
        """
        pass
    
    def gdpr_right_to_erasure(
        self,
        customer_id: str
    ):
        """
        Delete all data for customer (GDPR compliance)
        
        Process:
        1. Identify all metrics/logs containing customer_id
        2. Delete from:
           - Active database
           - Backups
           - Archives (if possible)
           - Audit logs (may be required for legal reasons)
        3. Log deletion request (required by GDPR)
        4. Verify complete deletion
        
        Complication:
        - Some audit logs required by law
        - Can only anonymize, not delete
        - Provide proof of deletion
        """
        pass
```

**Requirements**:
- [ ] Raw metrics: 30-day retention
- [ ] Aggregated metrics: 365-day retention
- [ ] RCA reports: 365-day retention
- [ ] Audit logs: 7-year retention
- [ ] Automatic purging at scheduled times
- [ ] Backup before deletion
- [ ] Secure deletion (overwrite with random data)
- [ ] GDPR right-to-erasure support

---

### 2.3 Model Safety & Validation

#### **2.3.1 Model Governance**

**Requirement**: Ensure models are validated before deployment

**Implementation**:

```python
class ModelGovernance:
    """
    Governance framework for RCA models
    """
    
    def model_validation(
        self,
        model: LSTMAutoencoder,
        test_scenarios: List[TestScenario]
    ) -> ModelValidationReport:
        """
        Validate model before deployment
        
        Validation checks:
        1. Performance Metrics
           - F1-score > 0.77 (80% precision, 75% recall)
           - False positive rate < 5%
           - False negative rate < 25%
           - Evaluated on held-out test set
        
        2. Edge Cases
           - Single-metric failure
           - Multiple simultaneous failures
           - Cascading failures
           - Rare failure types
           - Sensor noise/gaps
        
        3. Adversarial Testing
           - Can model be fooled by crafted data?
           - What if metric values reversed?
           - What if all metrics spiked?
           - What if metrics frozen?
        
        4. Fairness & Bias
           - Does model work equally well for all services?
           - Does it exhibit service bias?
           - Fair treatment of rare vs common failures?
        
        5. Interpretability
           - Can engineers understand why model makes decisions?
           - Are top-5 important features reasonable?
           - Do SHAP values match domain intuition?
        
        Output: Report with pass/fail on each check
        """
        pass
    
    def continuous_monitoring(
        self,
        model: LSTMAutoencoder,
        predictions: List[Prediction],
        ground_truth: List[Diagnosis]
    ) -> ModelPerformanceReport:
        """
        Monitor model performance in production
        
        Metrics tracked:
        1. Accuracy Drift
           - Weekly F1-score
           - Alert if drops >5%
           
        2. Data Drift
           - Are current metrics distribution matching training distribution?
           - Kullback-Leibler divergence test
           - Alert if KL divergence > 0.5
           
        3. Prediction Drift
           - Are model outputs changing?
           - Entropy of predictions
           - Alert if entropy spike
        
        4. False Positive Rate
           - Ratio of incorrect diagnoses
           - Track per-service
           - Alert if >5%
        
        Automated actions:
        - Alert when drift detected
        - Recommend retraining
        - Auto-trigger retraining if drift severe
        - Rollback to previous model if new model worse
        
        Frequency: Daily metric calculation, weekly review
        """
        pass
    
    def model_versioning(
        self,
        model: LSTMAutoencoder,
        version_name: str,
        training_date: datetime,
        performance_metrics: Dict
    ):
        """
        Maintain version history of models
        
        Versioning scheme:
        - v1.0.0: LSTM anomaly detection baseline
        - v1.0.1: Bug fix in preprocessing
        - v1.1.0: Added Transformer variant
        - v2.0.0: Major retraining with new data
        
        For each version track:
        - Model weights + architecture
        - Training data characteristics
        - Performance metrics
        - Known issues
        - Deployment history
        
        Enables:
        - Rolling back to previous version
        - Comparing performance across versions
        - Understanding which version deployed where
        - A/B testing new versions
        """
        pass
    
    def model_registry(self):
        """
        Central registry of all models
        
        Example entry:
        {
            'model_id': 'rca-lstm-v2.3.1',
            'type': 'LSTMAutoencoder',
            'status': 'production',
            'deployed_at': '2024-02-23T10:00:00Z',
            'deployed_by': 'ml-team-lead',
            'training_date': '2024-02-20',
            'training_data_size': 100000,
            'performance': {
                'f1_score': 0.82,
                'precision': 0.87,
                'recall': 0.78,
                'false_positive_rate': 0.03
            },
            'known_issues': [
                'Struggles with cascading failures >5 steps',
                'May miss subtle memory leaks'
            ],
            'previous_version': 'rca-lstm-v2.3.0',
            'next_scheduled_retraining': '2024-03-23',
            'owner': 'ml-team@company.com'
        }
        """
        pass
```

**Requirements**:
- [ ] All models validated before production deployment
- [ ] F1-score > 0.77, false positive rate < 5%
- [ ] Continuous monitoring for performance drift
- [ ] Alert if F1-score drops >5% in production
- [ ] Model versioning and rollback capability
- [ ] Central model registry
- [ ] Retraining scheduled monthly

---

#### **2.3.2 Causal Assumption Validation**

**Requirement**: Verify causal inference assumptions hold

**Implementation**:

```python
class CausalAssumptionValidator:
    """
    Validate that causal inference assumptions are satisfied
    """
    
    def validate_assumptions(
        self,
        data: pd.DataFrame,
        graph: nx.DiGraph
    ) -> CausalAssumptionReport:
        """
        Check all causal assumptions before inferring causation
        
        Assumptions:
        1. Acyclicity: No circular dependencies
           - Test: Topological sort
           - Fail action: Remove cycle edges; reduce confidence
           
        2. No unmeasured confounders: All causally relevant variables measured
           - Test: Check residual correlations
           - Fail action: Add missing variable; flag in confidence
           
        3. Causal sufficiency: No selection bias
           - Test: Verify data collection complete
           - Fail action: Warn about potential bias
           
        4. Temporal precedence: Cause before effect
           - Test: Verify timestamps
           - Fail action: Remove edge; flag as assumption violation
           
        5. Stationarity: Statistical properties don't change over time
           - Test: Augmented Dickey-Fuller test
           - Fail action: Recommend detrending; reduce confidence
        
        Output: {
            'all_assumptions_met': bool,
            'violations': [list of violations],
            'confidence_adjustment': float,  # 0.8-1.0
            'remedies': [recommended fixes]
        }
        """
        pass
    
    def adf_test_stationarity(
        self,
        time_series: np.ndarray,
        metric_name: str
    ) -> StationarityResult:
        """
        Augmented Dickey-Fuller test for stationarity
        
        Null hypothesis: Time series has unit root (non-stationary)
        Alternative: Time series is stationary
        
        If p-value < 0.05: Reject null hypothesis, series is stationary
        If p-value >= 0.05: Series likely non-stationary
        
        Non-stationary examples:
        - Memory leak (upward trend)
        - Changing traffic patterns (shift in mean)
        
        Remedy for non-stationary:
        - Differencing: Y_t = X_t - X_{t-1}
        - Detrending: Remove trend via polynomial fit
        - Log transform: Stabilize variance
        
        Return: (is_stationary, p_value, remedies)
        """
        from statsmodels.tsa.stattools import adfuller
        
        result = adfuller(time_series, autolag='AIC')
        
        is_stationary = result[1] < 0.05  # p-value
        
        remedies = []
        if not is_stationary:
            remedies = [
                'Apply differencing: Y[t] = X[t] - X[t-1]',
                'Apply log transformation for variance stabilization',
                'Detrend using polynomial fitting',
                'Reduce confidence in causal inference results'
            ]
        
        return StationarityResult(
            is_stationary=is_stationary,
            p_value=result[1],
            remedies=remedies
        )
    
    def detect_hidden_confounders(
        self,
        causal_graph: nx.DiGraph,
        data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Identify likely hidden confounders
        
        Heuristic: If two variables highly correlated but no edge between them
        in causal graph, likely hidden confounder
        
        Example:
        - Observed: CPU spike AND memory increase (both ~0.8 score)
        - No direct edge between CPU and memory
        - Likely confounder: New process consuming both CPU and memory
        
        Process:
        1. Calculate correlation matrix
        2. For each pair with high correlation (>0.7)
        3. Check if edge exists in causal graph
        4. If not, likely hidden confounder
        
        Output: {(var1, var2): confounder_likelihood (0-1)}
        """
        pass
    
    def check_temporal_precedence(
        self,
        causal_edges: List[Tuple[str, str]],
        detection_times: Dict[str, int]
    ) -> Dict[Tuple[str, str], bool]:
        """
        Verify cause precedes effect in time
        
        Rule: For edge (X → Y), must have:
        detection_time[X] <= detection_time[Y]
        
        If violated: Remove edge, reduce confidence
        
        Latency check:
        Also verify latency is reasonable
        - Within-process latency: <100ms
        - Service-to-service: <5 seconds
        - Distributed system: <10 minutes
        
        Output: {(cause, effect): is_valid}
        """
        pass
```

**Requirements**:
- [ ] Validate all causal assumptions before inferring causation
- [ ] Flag assumption violations with severity level
- [ ] Reduce confidence score if assumptions violated
- [ ] Provide remedies (detrending, etc.)
- [ ] Test for hidden confounders

---

### 2.4 Robustness & Graceful Degradation

#### **2.4.1 Failure Handling**

**Requirement**: System degrades gracefully under failure conditions

**Implementation**:

```python
class RobustnessFramework:
    """
    Handle failures gracefully without cascading
    """
    
    def handle_missing_metrics(
        self,
        metrics: Dict[str, any],
        required_metrics: Set[str]
    ) -> AnalysisResult:
        """
        Gracefully handle missing metrics
        
        Scenarios:
        1. Missing 1 metric: Proceed with remaining (confidence -10%)
        2. Missing 5+ metrics: Proceed but flag uncertainty (-30%)
        3. Missing >50% of metrics: Cannot proceed, return error
        
        Strategy:
        - Don't fail entirely
        - Analyze what you can
        - Clearly communicate limitations
        - Flag missing data in report
        
        Example:
        "Analysis completed with {N}% confidence.
        Missing metrics: [list].
        Results valid only for evaluated metrics."
        """
        pass
    
    def handle_model_inference_failure(
        self,
        metric: str,
        error: Exception
    ) -> FallbackPrediction:
        """
        Fallback if ML model fails
        
        Failure scenarios:
        1. CUDA out of memory: Run on CPU (slower)
        2. Model file corrupted: Use previous version
        3. Input shape mismatch: Resample to expected shape
        4. NaN values: Replace with baseline
        
        Fallback strategy:
        1. Try original inference
        2. If fails, try simpler method (statistical)
        3. If fails, use rule-based thresholds
        4. If fails, return "unknown" with explanation
        
        Never propagate error to user; always return result
        """
        pass
    
    def handle_resource_exhaustion(
        self,
        resource: str,  # 'cpu', 'memory', 'disk'
        utilization: float
    ):
        """
        Handle high resource usage
        
        Actions:
        1. CPU >90%: Cancel non-critical tasks, prioritize core analysis
        2. Memory >80%: Load data in chunks instead of all-at-once
        3. Disk >95%: Delete temporary files; archive old data
        
        Monitoring:
        - Alert if resources trending high
        - Implement resource quotas
        - Queue tasks when resources limited
        
        Graceful degradation:
        - Reduce analysis window (30 days → 7 days)
        - Reduce metric dimensionality
        - Increase analysis latency but maintain correctness
        """
        pass
    
    def handle_timeout(
        self,
        operation: str,
        timeout_seconds: int
    ) -> PartialResult:
        """
        Return partial results if analysis timeout
        
        Timeouts:
        - Model inference: 5 minutes
        - Causal inference: 10 minutes
        - Report generation: 2 minutes
        
        Strategy:
        - Process results so far
        - Flag what's incomplete
        - Show what you have
        - Suggest user wait and retry
        
        Example:
        "Analysis timeout after 10 minutes.
        Found 5 root causes (out of estimated 8).
        Top cause: {best_guess} with {confidence}% confidence.
        Retry for more complete analysis."
        """
        pass
    
    def circuit_breaker_pattern(
        self,
        failing_component: str,
        failure_threshold: int = 5,
        timeout_seconds: int = 300
    ):
        """
        Prevent cascading failures via circuit breaker
        
        States:
        - CLOSED: Component working, requests pass through
        - OPEN: Component failing, requests blocked immediately
        - HALF_OPEN: Component recovered, test requests allowed
        
        Rules:
        - After 5 consecutive failures: OPEN
        - After OPEN for 5 minutes: try HALF_OPEN
        - Successful request in HALF_OPEN: return to CLOSED
        
        Components:
        - Prometheus connection
        - CloudWatch connection
        - Model inference
        - Database connection
        
        Prevents:
        - System waiting for unresponsive component
        - Cascading timeouts
        - Resource exhaustion from retries
        """
        pass
    
    def rate_limiting(
        self,
        client_id: str,
        max_requests_per_minute: int = 10,
        max_analysis_per_hour: int = 100
    ) -> bool:
        """
        Prevent abuse via rate limiting
        
        Limits:
        - Per user: 10 analysis requests/minute
        - Per user: 100 analysis/hour
        - Per IP: 100 requests/minute
        - Per service: 1000 requests/hour
        
        Enforcement:
        - Token bucket algorithm
        - Return 429 (Too Many Requests) when exceeded
        - Provide retry-after header
        
        Protects against:
        - Accidental overwhelming (someone debugging with script loop)
        - Malicious DoS attack
        - Resource exhaustion
        """
        pass
```

**Requirements**:
- [ ] System handles missing metrics gracefully
- [ ] Fallback for ML model failures
- [ ] Resource exhaustion detection and mitigation
- [ ] Timeout protection with partial results
- [ ] Circuit breaker pattern for external dependencies
- [ ] Rate limiting per user/service

---

#### **2.4.2 Data Quality Checks**

**Requirement**: Detect and flag suspicious data patterns

**Implementation**:

```python
class DataQualityFramework:
    """
    Detect data quality issues that could lead to false diagnoses
    """
    
    def detect_suspicious_patterns(
        self,
        metrics: pd.DataFrame
    ) -> List[DataQualityConcern]:
        """
        Identify data patterns indicating problems
        """
        concerns = []
        
        # Pattern 1: Frozen metric
        # Same value for hours = likely sensor failure
        for col in metrics.columns:
            unique_count_last_hour = metrics[col].tail(12).nunique()
            if unique_count_last_hour <= 1:
                concerns.append(DataQualityConcern(
                    severity='WARNING',
                    metric=col,
                    issue='Metric frozen (same value for 1 hour)',
                    impact='Cannot detect anomalies in frozen metric',
                    remedy='Check sensor/collector health'
                ))
        
        # Pattern 2: Sudden discontinuity
        # Metric jumps from 50 to 0 to 50 = likely collection issue
        for col in metrics.columns:
            diffs = metrics[col].diff().abs()
            mean_diff = diffs.mean()
            recent_diffs = diffs.tail(12)
            
            spikes = (recent_diffs > 5 * mean_diff).sum()
            if spikes > 3:  # More than 25% of readings are spikes
                concerns.append(DataQualityConcern(
                    severity='WARNING',
                    metric=col,
                    issue=f'High discontinuity rate: {spikes}/12 readings',
                    impact='Unreliable anomaly detection',
                    remedy='Investigate metric source'
                ))
        
        # Pattern 3: Out-of-range values
        # Percentage should be 0-100, but measured 150
        for col in metrics.columns:
            if col.endswith('_percent') or col.endswith('_rate'):
                if (metrics[col] < -10).any() or (metrics[col] > 110).any():
                    concerns.append(DataQualityConcern(
                        severity='ERROR',
                        metric=col,
                        issue=f'Out-of-range values: {metrics[col].min()}-{metrics[col].max()}',
                        impact='Invalid analysis results',
                        remedy='Correct metric collection'
                    ))
        
        # Pattern 4: Sudden collection gap
        # Metrics present for 30 days then suddenly missing
        if metrics.isna().sum().sum() > 0:
            last_available = {}
            for col in metrics.columns:
                non_null_indices = metrics[col].notna()
                if non_null_indices.any():
                    last_available[col] = non_null_indices[::-1].argmax()
            
            # If recently missing
            if any(last_available.values()):
                earliest_gap = min(last_available.values())
                if earliest_gap < 12:  # Within last hour
                    concerns.append(DataQualityConcern(
                        severity='WARNING',
                        metric=f'{sum(v < 12 for v in last_available.values())} metrics',
                        issue='Recent data collection gap',
                        impact='Analysis incomplete for recent period',
                        remedy='Check collector/connectivity'
                    ))
        
        return concerns
    
    def validate_against_schema(
        self,
        metrics: pd.DataFrame,
        schema: Dict
    ) -> List[ValidationError]:
        """
        Validate metrics match expected schema
        
        Schema example:
        {
            'cpu_utilization': {
                'type': 'float',
                'min': 0.0,
                'max': 100.0,
                'unit': 'percent',
                'latency_p95_ms': <100
            },
            'api_latency_p99_ms': {
                'type': 'float',
                'min': 0.0,
                'max': 60000.0,  # Max reasonable latency
                'unit': 'milliseconds'
            }
        }
        
        Validation:
        1. Check column exists
        2. Check data type
        3. Check range
        4. Check sampling frequency
        5. Check for expected periodicity
        
        Return: List of validation errors
        """
        pass
    
    def cross_metric_sanity_check(
        self,
        metrics: pd.DataFrame
    ) -> List[SanityCheckWarning]:
        """
        Cross-validate relationships between metrics
        
        Sanity checks:
        1. CPU vs API Latency
           - Should be somewhat correlated (r > 0.3)
           - If completely uncorrelated: suspicious
           
        2. Memory vs GC Time
           - Heap pressure should correlate with GC
           - Missing correlation: possible monitoring issue
           
        3. Request Count vs Latency
           - More load usually → higher latency
           - Inverse relationship: suspicious
        
        4. Error Rate vs Latency
           - When latency high, error rate usually up
           - Inverse correlation: suspicious
        
        Process:
        1. Calculate expected correlations
        2. Compare to observed correlations
        3. Flag significant deviations
        """
        pass
    
    def anomaly_validation(
        self,
        detected_anomalies: Dict[str, float],
        metrics_values: Dict[str, float],
        baselines: Dict[str, float]
    ) -> Dict[str, AnomalyValidation]:
        """
        Validate each detected anomaly makes sense
        
        Checks:
        1. Magnitude: Is deviation large enough?
           - Baseline 50%, Measured 51% = 2% change
           - Should this trigger anomaly? Unlikely (threshold should be >10%)
        
        2. Consistency: Do multiple detection methods agree?
           - LSTM: Anomaly score 0.92
           - Statistical: Anomaly score 0.15
           - Disagreement: Lower confidence
        
        3. Plausibility: Does the anomaly make physical sense?
           - CPU 0% and Memory 100% = impossible
           - Both possible separately, but together = suspicious
        
        Output: {metric: {
            'is_valid': bool,
            'confidence_adjustment': float,
            'warnings': [list of concerns]
        }}
        """
        pass
```

**Requirements**:
- [ ] Detect frozen metrics
- [ ] Flag discontinuities in data
- [ ] Validate against metric schema
- [ ] Cross-metric sanity checking
- [ ] Anomaly plausibility validation
- [ ] Clear warnings in reports

---

### 2.5 Preventing False Diagnoses

#### **2.5.1 False Positive Mitigation**

**Requirement**: Minimize incorrect root cause identifications

**Problem**: Incorrectly identifying the root cause can lead to:
- Wasting engineering time investigating wrong area
- Missing the real problem while it continues
- Applying wrong remediation (e.g., restarting wrong service)
- Escalating incident inappropriately

**Solution**:

```python
class FalsePositiveMitigation:
    """
    Reduce false positives through multiple strategies
    """
    
    def multi_method_consensus(
        self,
        granger_result: CausalityResult,
        pc_algorithm_result: CausalityResult,
        domain_knowledge: Dict,
        temporal_check: bool
    ) -> bool:
        """
        Require agreement between multiple methods
        
        Strategy: Voting system
        
        Granger Causality Vote:
        - If p-value < 0.05: +1 vote for causality
        - If temporal lag reasonable: +1 bonus
        
        PC Algorithm Vote:
        - If edge in learned graph: +1 vote
        - If edge strength high: +1 bonus
        
        Domain Knowledge Vote:
        - If causality aligns with documented architecture: +1 vote
        - If contradicts known patterns: -1 vote
        
        Temporal Precedence Vote:
        - If cause precedes effect in time: +1 vote
        - If temporal lag reasonable: +1 bonus
        
        Voting outcome:
        - 4+ votes: HIGH confidence (accept)
        - 3 votes: MEDIUM confidence (review)
        - <3 votes: LOW confidence (reject or flag)
        
        Prevents: Algorithms agreeing on wrong answer
        """
        pass
    
    def alternative_hypotheses_generation(
        self,
        top_root_cause: str,
        causal_graph: nx.DiGraph,
        anomaly_scores: Dict[str, float]
    ) -> List[AlternativeHypothesis]:
        """
        Generate alternative explanations to challenge top answer
        
        Process:
        1. Find top candidate (e.g., db_latency)
        2. Ask: What if it's not db_latency?
        3. Generate alternatives:
           a) Different root cause (e.g., connection pool)
           b) Multiple simultaneous causes
           c) Hidden confounder (unmeasured variable)
           d) False correlation / noise
        4. Test each hypothesis
        5. Compare confidence levels
        
        Output: [{
            'hypothesis': 'Connection pool was always near capacity, surge exposed vulnerability',
            'likelihood': 0.3,
            'distinguishing_test': 'Check if connection creation errors in logs',
            'confidence_impact': 'Reduces primary hypothesis confidence from 0.87 to 0.75'
        }]
        
        Prevents: Tunnel vision on first hypothesis
        """
        pass
    
    def contradiction_detection(
        self,
        causal_chain: List[str],
        metrics_timeline: Dict[str, List[float]]
    ) -> List[Contradiction]:
        """
        Identify internal contradictions in causal chain
        
        Example contradictions:
        1. Chain says "db latency → api timeout", but API latency stable while db latency spiked
        2. Chain says "memory leak → OOM", but memory usage plateaued
        3. Chain says "X caused Y", but X still present while Y recovered
        
        Check for each:
        1. Temporal consistency: Does timeline match narrative?
        2. Magnitude consistency: Are effects proportional to causes?
        3. Recovery consistency: When root cause fixed, do symptoms resolve?
        
        Output: [{
            'contradiction': 'Root cause "db_latency" detected at 15:30, but API latency was normal until 16:00',
            'severity': 'HIGH',
            'implication': 'Root cause identification may be incorrect'
        }]
        
        Prevents: Logically inconsistent diagnoses
        """
        pass
    
    def baseline_comparison(
        self,
        current_detection: str,
        historical_failures: List[FailureRecord]
    ) -> SimilarityAnalysis:
        """
        Compare to similar historical failures
        
        Process:
        1. Find historical failures with similar symptoms
        2. Compare:
           - Which metrics deviated?
           - Root causes found?
           - Time-to-resolution?
        3. Check if current matches known patterns
        4. Highlight unusual aspects
        
        Example:
        "This failure has high similarity to incident INC-2023-042:
        - Both: DB connections spike
        - Both: API latency increase
        - INC-2023-042 root cause: Connection leak in new deployment
        - Current root cause hypothesis: Database slow query
        
        Recommendation: Check recent deployments in addition to database"
        
        Prevents: Missing obvious pattern that happened before
        """
        pass
    
    def false_positive_feedback_loop(
        self,
        diagnosis: RootCauseAnalysis,
        validation_result: DiagnosisValidation
    ):
        """
        Learn from false positives to improve future detection
        
        Process:
        1. After diagnosis validated as wrong (false positive):
           a. Log incorrect diagnosis
           b. Store true root cause (if found)
           c. Analyze why model got it wrong
        
        2. Retrain model with this negative example:
           a. Add case to training data
           b. Reweight model parameters
           c. Increase confidence threshold
        
        3. Update anomaly thresholds if needed
        4. Notify engineer team
        
        Prevents: Repeated same false positive
        """
        pass
```

**Requirements**:
- [ ] Require multi-method consensus for root cause
- [ ] Generate and test alternative hypotheses
- [ ] Detect contradictions in causal chains
- [ ] Compare to historical similar failures
- [ ] Learning feedback loop for false positives
- [ ] False positive rate < 5%

---

#### **2.5.2 False Negative Mitigation**

**Requirement**: Minimize missed root causes

**Problem**: If system misses the true root cause:
- Engineers continue investigating wrong areas
- Real problem persists
- Incident not resolved
- False confidence in wrong diagnosis

**Solution**:

```python
class FalseNegativeMitigation:
    """
    Ensure true root causes are found
    """
    
    def coverage_assessment(
        self,
        metrics_available: Set[str],
        known_failure_modes: Dict
    ) -> CoverageReport:
        """
        Assess if system has metrics to detect known failure modes
        
        Failure mode checklist:
        - Memory leak: Need memory usage trends
        - Slow query: Need query latency metrics
        - Connection pool exhaustion: Need pool utilization
        - CPU spike: Need CPU metrics
        
        For each known failure mode:
        1. Check: Required metrics available?
        2. If missing: Flag as coverage gap
        3. Recommend: Which metrics to add
        
        Output: {
            'coverage_pct': 87,  # 87% of known failure modes covered
            'gaps': ['DNS failure - no DNS latency metric', ...],
            'recommendations': ['Add DNS probe metrics', ...]
        }
        """
        pass
    
    def sensitivity_analysis(
        self,
        anomaly_detection_thresholds: Dict[str, float]
    ) -> SensitivityReport:
        """
        Ensure thresholds aren't too high (missing anomalies)
        
        Process:
        1. Test: Lower threshold by 10%, 20%, 30%
        2. Check: How many historical failures would be caught?
        3. Monitor: False positive rate at each threshold
        
        Trade-off:
        - Higher threshold: Lower false positives, more false negatives
        - Lower threshold: Higher false positives, fewer false negatives
        
        Recommendation:
        - Aim for asymmetry: 5% false positives, 2% false negatives
        - Missing real problems worse than false alarms
        
        Output: {
            'current_threshold': 0.7,
            'estimated_false_negatives': '3-5%',
            'estimated_false_positives': '2%',
            'recommendation': 'Consider lowering threshold to 0.65'
        }
        """
        pass
    
    def active_search_for_root_cause(
        self,
        detected_anomalies: Dict[str, float],
        causal_graph: nx.DiGraph
    ) -> List[PotentialRootCause]:
        """
        Don't just rank existing anomalies; search for hidden causes
        
        Searches:
        1. Upstream search: Find causes of causes
           - If db_latency high, look for what caused db_latency
           - Trace back chain
        
        2. Parallel search: Find related metrics that deviated
           - If CPU high, check if disk I/O also high (could be both)
           - Correlation analysis
        
        3. Hidden metric search: Infer unobserved variables
           - If API latency up but no monitored bottleneck,
             might be unmonitored external service
           - Hypothesize missing metrics
        
        4. Temporal search: Look back further in time
           - Maybe root cause appeared before first noticed symptom
           - Extend analysis window
        
        Output: List of potential root causes including:
        - Observable anomalies
        - Inferred hidden causes
        - Hypothesized missing metrics
        """
        pass
    
    def ensemble_detection_methods(
        self,
        metrics: pd.DataFrame
    ) -> EnsembleAnomalyResult:
        """
        Use multiple complementary anomaly detection methods
        
        Methods:
        1. LSTM reconstruction error
        2. Statistical z-score
        3. Isolation Forest
        4. One-class SVM
        5. LOF (Local Outlier Factor)
        6. Seasonal decomposition
        7. Time series forecasting
        
        Ensemble strategy:
        - Majority voting (≥4 methods agree)
        - Weighted voting (weight by method precision)
        - Meta-learner (learn combination)
        
        Benefit:
        - Different methods catch different anomalies
        - One method's false negative = another's true positive
        - More robust than single method
        
        Drawback:
        - Higher computational cost
        - More false positives (need thresholding)
        
        Trade: Worth it to avoid missing real problems
        """
        pass
    
    def explanatory_gap_analysis(
        self,
        diagnosis: RootCauseAnalysis,
        unexplained_anomalies: List[str]
    ) -> ExplanationGapReport:
        """
        Check if diagnosis explains all observed anomalies
        
        Example:
        - Diagnosis: "Database slow query"
        - Explains: API latency, error rate
        - Does NOT explain: Memory spike
        - Gap: What caused memory spike?
        
        Process:
        1. List all anomalies detected
        2. Check: Can root cause explain each?
        3. If anomaly unexplained:
           a. Either it's secondary effect (trace the chain)
           b. Or it's independent failure (separate root cause)
        
        Output: {
            'anomalies_explained': 7,
            'total_anomalies': 10,
            'unexplained': ['memory_spike', 'cache_miss_rate'],
            'interpretation': 'Suggests either:
                1. Missing causal links (update graph)
                2. Multiple independent failures (need multi-cause analysis)
                3. Hidden confounders affecting memory separately'
        }
        """
        pass
```

**Requirements**:
- [ ] Coverage assessment of available metrics
- [ ] Sensitivity analysis of anomaly thresholds
- [ ] Active search for hidden root causes
- [ ] Ensemble detection methods
- [ ] Explanation gap analysis
- [ ] False negative rate < 5%

---

### 2.6 Security Against Misuse

#### **2.6.1 Preventing Intentional Abuse**

**Threat**: Malicious actor could use RCA to:
- Hide failures (suppress alerts)
- Trigger false alarms (cause panic)
- Gain unauthorized system access
- Sabotage competitor or internal team

**Mitigation**:

```python
class MisusePreventionFramework:
    """
    Prevent intentional abuse of RCA system
    """
    
    def detect_suspicious_queries(
        self,
        query: AnalysisRequest,
        user_profile: UserProfile
    ) -> List[SuspiciousIndicator]:
        """
        Detect unusual or suspicious analysis requests
        
        Red flags:
        1. User requests analysis of service they don't own
        2. User requests analysis of specific failure mode right before it happens
        3. User repeatedly queries same service with different parameters
        4. Bulk export of raw metrics data
        5. Request during unusual hours for user
        6. Request from unusual geographic location
        
        Example:
        "User normally analyzes 'api-server', suddenly analyzing 'auth-service'.
        No access to auth-service. Flagged as suspicious."
        
        Actions:
        - Log and flag for security review
        - Request manager approval
        - Deny if flagged high-risk
        """
        pass
    
    def audit_trail_verification(
        self,
        diagnosis: RootCauseAnalysis,
        who_ran_it: str,
        when_run: datetime
    ) -> AuditRecord:
        """
        Create immutable audit record for accountability
        
        Record includes:
        - Who ran analysis (user ID)
        - When run (timestamp)
        - What was analyzed (service, metrics)
        - What was found (root cause)
        - Actions taken based on diagnosis
        - Validation of diagnosis (was it correct?)
        
        Immutability:
        - Stored with cryptographic hash
        - Hash chain: each record includes hash of previous
        - Cannot modify without detection
        
        Enables:
        - Detecting who sabotaged diagnosis
        - Replaying analysis history
        - Compliance audits
        
        Retention: 7 years (per regulation)
        """
        pass
    
    def honeypot_metrics(self):
        """
        Plant fake metrics to detect tampering
        
        Setup:
        - Add fake metric: "fake_cpu_always_normal"
        - Metric always returns 45% (normal range)
        - If suddenly spikes, detector working
        - If detector misses it, might be compromised
        
        Monitoring:
        - Alert if fake metric anomaly detected
        - Alert if fake metric goes missing
        - Alert if fake metric changes values
        
        Purpose:
        - Detect if someone tampering with metrics
        - Validate detector integrity
        - Early warning of system compromise
        """
        pass
    
    def result_validation_by_peer(
        self,
        diagnosis: RootCauseAnalysis,
        peer_reviewer: str
    ) -> ValidationReport:
        """
        Require peer review for critical diagnoses
        
        Triggers for mandatory peer review:
        - Diagnosis recommends major remediation (service restart)
        - Diagnosis identifies security vulnerability
        - Diagnosis involves accessing sensitive data
        - User has history of false diagnoses
        
        Peer review process:
        1. Peer independently analyzes same failure
        2. Compares to original diagnosis
        3. Flags any discrepancies
        4. Approves or rejects diagnosis
        5. Provides written justification
        
        Prevents:
        - Single person making incorrect high-impact decisions
        - Sabotage via fake diagnosis
        - Mistakes due to fatigue or bias
        """
        pass
    
    def detect_repeated_attack_patterns(self):
        """
        Identify if someone repeatedly triggering same false diagnosis
        
        Pattern:
        "User X has submitted 12 analyses in past 24 hours.
        All point to database team's service as root cause.
        All diagnoses later proven wrong.
        Possible targeted attack against database team."
        
        Actions:
        - Alert security team
        - Disable user's analysis capability
        - Investigate user's credentials/access
        - Notify database team
        """
        pass
```

**Requirements**:
- [ ] Detect suspicious analysis requests
- [ ] Immutable audit trail of all diagnoses
- [ ] Honeypot metrics to detect tampering
- [ ] Peer review for critical diagnoses
- [ ] Detection of repeated attack patterns
- [ ] Security team alerts for anomalies

---

#### **2.6.2 Preventing Accidental Harm**

**Threat**: Well-intentioned user could:
- Misunderstand diagnosis and apply wrong remediation
- Share sensitive failure information externally
- Recommend premature escalation

**Mitigation**:

```python
class AccidentalHarmPrevention:
    """
    Prevent honest mistakes from causing damage
    """
    
    def remediation_confirmation_required(
        self,
        recommendation: RemediationStep,
        impact_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    ) -> bool:
        """
        Require explicit confirmation before high-impact actions
        
        Impact levels:
        - LOW: "Check logs" - No confirmation needed
        - MEDIUM: "Increase timeout threshold" - Requires manager approval
        - HIGH: "Restart service" - Requires manager + on-call approval
        - CRITICAL: "Delete data" or "Database migration" - Requires CTO approval
        
        Confirmation process:
        1. Show recommendation
        2. Show expected impact (outage duration, affected users)
        3. Show rollback plan
        4. Require typed confirmation: "I understand this will..."
        5. Log who approved
        
        Prevents:
        - Accidentally clicking button and restarting prod database
        - Misunderstanding latency implications
        - One person making critical decisions alone
        """
        pass
    
    def expected_impact_estimation(
        self,
        remediation: RemediationStep
    ) -> ImpactEstimate:
        """
        Estimate impact before action taken
        
        Estimation includes:
        - Expected outage duration: 2-5 minutes
        - Estimated affected users: 5,000-10,000
        - Estimated affected transactions: $50k-100k value
        - Confidence in estimate: 70%
        
        Helps user make informed decision:
        "This fix will cause 2-minute outage. Worth it for 7-hour fix?"
        
        Prevents:
        - Acting on diagnosis without understanding consequences
        - Disproportionate response (shutting down everything to fix small issue)
        """
        pass
    
    def data_sensitivity_flagging(
        self,
        report: RCAReport,
        contains_sensitive: List[str]
    ) -> None:
        """
        Flag and redact sensitive data in reports
        
        Sensitive data:
        - Customer names/IDs
        - Internal IP addresses
        - API keys or tokens
        - Personal information (names, emails)
        - Financial data
        - Health information (HIPAA)
        
        Flagging:
        1. Automatically redact from default report
        2. Flag where redaction occurred
        3. Require explicit request to see full details
        4. Log who requested unredacted data
        
        Example:
        Before: "Query from user 12345 (John Smith, john@example.com) failed with..."
        After: "Query from user [REDACTED] ([REDACTED], [REDACTED]) failed with..."
        
        Prevents:
        - Accidentally sharing PII in incident report
        - Security vulnerability from exposed credentials
        - Compliance violations (GDPR, HIPAA, PCI-DSS)
        """
        pass
    
    def escalation_sanity_check(
        self,
        recommendation: Escalation,
        current_severity: int,
        escalation_target: str
    ) -> bool:
        """
        Prevent over-escalation
        
        Sanity checks:
        1. Is escalation proportional to severity?
           - LOW severity → escalating to CTO: suspicious
           - CRITICAL severity → only notifying team lead: suspicious
        
        2. Is escalation target appropriate?
           - Database issue → escalate to SRE team, not HR
           - Vendor issue → escalate to vendor manager, not database team
        
        3. Is escalation premature?
           - Has immediate team had time to investigate? (usually needs 5-10 min)
           - Have we gathered minimum diagnostic info?
           - Is this known issue with known mitigation?
        
        Process:
        1. Check recommendation against rules
        2. If suspicious, ask: "Did you mean to escalate to {target}?"
        3. Provide context: "Previous similar issue resolved by {team} in {time}"
        4. Offer alternatives
        
        Prevents:
        - Boy-who-cried-wolf escalations
        - Waking up CTO for low-severity issues
        - Escalating to wrong team (wasted time)
        """
        pass
    
    def inference_confidence_communication(
        self,
        root_cause: str,
        confidence: float,
        confidence_interval: Tuple[float, float]
    ) -> str:
        """
        Clearly communicate confidence level and uncertainty
        
        Communication:
        "Root cause identified as {root_cause}
        Confidence: {confidence:.0%} (95% CI: {ci_lower:.0%}-{ci_upper:.0%})
        
        What this means:
        - {confidence:.0%}: We believe this is the real cause
        - {100-confidence:.0%}: It could be something else
        - Confidence interval: We're 95% sure true confidence is between {ci_lower:.0%} and {ci_upper:.0%}
        
        Recommendation strength:
        - If confidence >85%: Investigate this cause immediately
        - If confidence 70-85%: Investigate while also exploring alternatives
        - If confidence <70%: Treat as hypothesis only, not definitive diagnosis
        
        Common misunderstandings we're preventing:
        - DON'T: "It's 80% confident, so it's definitely the cause" ✗
        - DO: "It's 80% likely, but there's 20% chance of something else" ✓
        "
        
        Prevents:
        - Over-confident decisions
        - Ignoring uncertainty
        - False sense of certainty
        """
        pass
```

**Requirements**:
- [ ] Confirmation required for high-impact remediations
- [ ] Expected impact estimation before action
- [ ] Sensitive data flagging and redaction
- [ ] Escalation sanity checking
- [ ] Clear uncertainty communication

---

## 3. NON-FUNCTIONAL REQUIREMENTS

### 3.1 Performance

| Metric | Target | Priority |
|--------|--------|----------|
| Anomaly detection latency | <5 minutes | Critical |
| Causal inference latency | <10 minutes | Critical |
| Report generation latency | <2 minutes | High |
| Dashboard load time | <3 seconds | High |
| Concurrent users | 50+ | Medium |
| Metric ingestion throughput | 100k metrics/minute | High |

### 3.2 Scalability

- [ ] Support 100+ metrics simultaneously
- [ ] Scale to 1000+ services
- [ ] Horizontal scaling for inference workers
- [ ] Database indexes optimized for time-range queries

### 3.3 Availability

- [ ] 99.9% uptime SLA for core services
- [ ] Graceful degradation under load
- [ ] Automatic failover for database connections
- [ ] Health checks every 30 seconds

### 3.4 Maintainability

- [ ] Code coverage >80%
- [ ] Automated testing for all components
- [ ] Clear architecture documentation
- [ ] Runbook for common issues
- [ ] Change log for all updates

---

## 4. SUCCESS CRITERIA & EVALUATION

### 4.1 Technical Success Metrics

```python
class SuccessMetrics:
    """
    Objective measures of system success
    """
    
    # Accuracy metrics
    top_1_accuracy = 0.65  # Correct root cause in #1 spot
    top_3_accuracy = 0.85  # Correct root cause in top 3
    mean_reciprocal_rank = 0.72  # Average rank of correct answer
    
    # Detection metrics
    precision = 0.87  # When we say it's an anomaly, it usually is
    recall = 0.78  # We catch most real anomalies
    f1_score = 0.82  # Balanced measure
    
    # False positive/negative rates
    false_positive_rate = 0.03  # 3% of alerts are false
    false_negative_rate = 0.05  # We miss 5% of real failures
    
    # Latency metrics
    analysis_latency_p95 = 300  # seconds (5 minutes for 95th percentile)
    report_generation_latency_p95 = 120  # seconds (2 minutes)
    
    # Availability
    system_uptime = 0.999  # 99.9%
    
    # User satisfaction
    user_satisfaction = 0.80  # 80% of users find reports useful
```

### 4.2 Business Success Metrics

```python
class BusinessMetrics:
    """
    Impact on business operations
    """
    
    # Operational improvement
    mttr_reduction = 0.75  # Mean time to resolve reduced by 75%
        # Before RCA: 4 hours, After: 1 hour
    
    false_diagnosis_impact = 0.05  # 5% of diagnoses prove wrong
        # Cost per wrong diagnosis: 2 engineer-hours wasted
    
    # Customer impact
    incident_resolution_improvement = 0.60  # 60% faster resolution
    customer_satisfaction_improvement = 0.15  # 15% increase in satisfaction
    
    # Financial
    estimated_annual_savings = 250000  # dollars
        # Calculation:
        # 365 incidents/year * 4 hours saved/incident * $100/hour
        # = 365 * 4 * 100 = $146k base
        # + reduced error costs, faster feature time, reduced escalations
        # = $250k estimated total
    
    roi = 5.0  # $5 saved for every $1 spent on system
```

---

## 5. APPENDIX: FAILURE MODE EXAMPLES

### Example 1: Database Slow Query

```
FAILURE TIMELINE:
T=0:00   Database schema migration deployed (added composite index)
T=4:30   First anomaly: product_query_latency 50ms → 200ms
T=8:45   Connection pool utilization 65% → 95%
T=15:00  API error rate spikes to 15%
T=20:00  Full service outage declared

ROOT CAUSE ANALYSIS:
Primary: Database schema migration (87% confidence)
Evidence:
- Temporal precedence: Schema change → latency increase (4.5 hours later)
- Causal chain: Slow query → Connection exhaustion → API timeout
- Granger causality test p-value: 0.001
- PC algorithm: schema_change → query_latency → connection_pool → error_rate

CAUSAL CHAIN:
Schema migration
↓ (mechanism: Query planner chose inefficient index)
Slow queries (+400% latency)
↓ (mechanism: Connections held longer)
Connection pool exhaustion (95% utilization)
↓ (mechanism: New requests wait in queue, then timeout)
API request timeouts
↓ (mechanism: Downstream services fail after timeout)
User-facing errors (15% error rate)

REMEDIATION:
Immediate: Rollback schema migration (5 minutes)
Short-term: Force query optimizer to use old index (immediate)
Long-term: Benchmark query performance before deployment
```

### Example 2: Memory Leak

```
FAILURE TIMELINE:
T=0:00   Code deployment v2.14.3 (WebSocket handler changes)
T=4:00   First anomaly detected: Memory 1.5GB → 1.6GB
T=24:00  Memory 2.1GB (continues increasing)
T=48:00  Memory 3.8GB → OOM kill → service restarts
T=48:30  Same OOM kill cycle repeats

ROOT CAUSE ANALYSIS:
Primary: Memory leak in new WebSocket handler (91% confidence)
Evidence:
- LSTM trend detection: Monotonic increase over 48 hours
- Linear regression: R² = 0.98, positive slope
- Correlation with deployment: 0.89 (4-hour lag)
- Mechanism: Connections not released from pool

TECHNICAL DETAILS:
- Leaked connections per client: 2MB buffer
- Connection leak rate: ~40 connections/hour
- After 48 hours: 1,800 leaked connections = 3.6GB
- GC unable to free (connections held in memory)

CAUSAL CHAIN:
Buggy code deployed
↓ (WebSocket disconnect handler missing connection.close())
Connections leak (retained in pool)
↓ (mechanisms: Each holds 2MB buffer)
Memory usage increases linearly
↓ (mechanisms: GC can't free; heap pressure increases)
GC overhead increases
↓ (CPU spent on garbage collection)
Request processing latency increases
↓ (timeout cascade)
Queue backlog
↓
Service crash

REMEDIATION:
Immediate: Rollback to v2.14.2 (2 minutes)
Fix: Add connection.close() in disconnect handler (code review, 30 min)
Testing: Memory leak detection tests added
Deployment: Canary deployment to 5% traffic first
```

---

## 6. APPENDIX: ASSUMPTIONS & CONSTRAINTS

### Assumptions

1. **Data Availability**: 30+ days of healthy operational data available
2. **Metric Coverage**: Comprehensive monitoring of all critical components
3. **Temporal Ordering**: Timestamps accurate to within 1 minute
4. **No Hidden Variables**: All causally relevant metrics are measured
5. **Linear Relationships**: Metric relationships approximately linear
6. **Acyclic Dependencies**: No circular system dependencies

### Constraints

1. **Analysis Latency**: Cannot provide sub-5-minute analysis (batch processing)
2. **New Services**: Cannot analyze services with <30 days history
3. **Cascading Failures**: Cannot reliably diagnose >5 simultaneous anomalies
4. **External Failures**: Cannot detect failures outside monitored infrastructure
5. **Configuration Complexity**: Requires manual setup for each monitored system
6. **Expertise Required**: Results should be reviewed by domain experts

---

## 7. FUTURE ENHANCEMENTS (v2.0+)

- [ ] Automated remediation with approval workflows
- [ ] Predictive failure detection (before impact)
- [ ] Multi-tenant support with data isolation
- [ ] Real-time streaming analysis (sub-minute latency)
- [ ] Integration with incident management systems
- [ ] Mobile alerts for critical diagnoses
- [ ] AI-powered runbook recommendations
- [ ] Causal inference confidence intervals (Bayesian)
- [ ] Custom ML model training per-service
- [ ] Cross-organization failure correlation

---

**Document Version**: 1.0  
**Last Updated**: February 23, 2024  
**Status**: Final - Ready for Implementation