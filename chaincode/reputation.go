package main

import (
	"encoding/json"
	"fmt"
	"math"
	"sort"

	"github.com/hyperledger/fabric-contract-api-go/v2/contractapi"
)

const (
	initialReputation       int64 = 8000
	maximumReputation       int64 = 10000
	trustedReputation       int64 = 7000
	blockedReputation       int64 = 5000
	previousScoreWeight     int64 = 80
	warmupRounds                  = 2
	consecutiveAnomalyLimit       = 2
	minimumDetectorClients        = 3
	statusTrusted                 = "TRUSTED"
	statusWatch                   = "WATCH"
	statusBlocked                 = "BLOCKED"
)

type ClientReputation struct {
	DocType              string `json:"doc_type"`
	ExperimentID         int    `json:"experiment_id"`
	ClientID             int    `json:"client_id"`
	Score                int64  `json:"score"`
	Status               string `json:"status"`
	Assessments          int    `json:"assessments"`
	Anomalies            int    `json:"anomalies"`
	ConsecutiveAnomalies int    `json:"consecutive_anomalies"`
	LastRoundID          int    `json:"last_round_id"`
}

type ClientAssessment struct {
	DocType              string `json:"doc_type"`
	RoundID              int    `json:"round_id"`
	ExperimentID         int    `json:"experiment_id"`
	ClientID             int    `json:"client_id"`
	Distance             int64  `json:"distance"`
	Threshold            int64  `json:"threshold"`
	ComparableValues     int    `json:"comparable_values"`
	Assessed             bool   `json:"assessed"`
	Anomalous            bool   `json:"anomalous"`
	Included             bool   `json:"included"`
	PreviousScore        int64  `json:"previous_score"`
	RoundScore           int64  `json:"round_score"`
	NewScore             int64  `json:"new_score"`
	Status               string `json:"status"`
	ConsecutiveAnomalies int    `json:"consecutive_anomalies"`
}

type ReputationReport struct {
	DocType        string             `json:"doc_type"`
	RoundID        int                `json:"round_id"`
	ExperimentID   int                `json:"experiment_id"`
	Sequence       int                `json:"sequence"`
	Warmup         bool               `json:"warmup"`
	DetectionUsed  bool               `json:"detection_used"`
	MedianDistance int64              `json:"median_distance"`
	MAD            int64              `json:"mad"`
	Threshold      int64              `json:"threshold"`
	Assessments    []ClientAssessment `json:"assessments"`
}

func assessPrototypeReputations(
	ctx contractapi.TransactionContextInterface,
	round *Round,
	records []PrototypeRecord,
) ([]ClientAssessment, []ClientReputation, *ReputationReport, error) {
	references, eligibleClasses := robustReferences(round, records)
	distances := make([]int64, len(records))
	comparable := make([]int, len(records))
	activeDistances := make([]int64, 0, len(records))
	for index, record := range records {
		differences := make([]int64, 0)
		for classID := 0; classID < round.NumClasses; classID++ {
			if !eligibleClasses[classID] || record.Counts[classID] == 0 {
				continue
			}
			offset := classID * round.Dimension
			for dimensionID := 0; dimensionID < round.Dimension; dimensionID++ {
				position := offset + dimensionID
				differences = append(differences, absoluteDifference(record.Values[position], references[position]))
			}
		}
		comparable[index] = len(differences)
		if len(differences) > 0 {
			distances[index] = medianInt64(differences)
			activeDistances = append(activeDistances, distances[index])
		}
	}

	detectionUsed := len(activeDistances) >= minimumDetectorClients
	medianDistance, mad, threshold := int64(0), int64(0), int64(0)
	if detectionUsed {
		medianDistance = medianInt64(activeDistances)
		deviations := make([]int64, len(activeDistances))
		for index, distance := range activeDistances {
			deviations[index] = absoluteDifference(distance, medianDistance)
		}
		mad = medianInt64(deviations)
		margin := saturatingMultiply(mad, 3)
		minimumMargin := round.Scale / 100
		if minimumMargin < 1 {
			minimumMargin = 1
		}
		if margin < minimumMargin {
			margin = minimumMargin
		}
		threshold = saturatingAdd(medianDistance, margin)
	}

	assessments := make([]ClientAssessment, len(records))
	reputations := make([]ClientReputation, len(records))
	for index, record := range records {
		reputation, err := loadClientReputation(ctx, round.ExperimentID, record.ClientID)
		if err != nil {
			return nil, nil, nil, err
		}
		previousScore := reputation.Score
		assessed := detectionUsed && comparable[index] > 0
		anomalous := assessed && distances[index] > threshold
		roundScore := previousScore
		if assessed {
			reputation.Assessments++
			if anomalous {
				roundScore = 0
				reputation.Anomalies++
				reputation.ConsecutiveAnomalies++
			} else {
				roundScore = maximumReputation
				reputation.ConsecutiveAnomalies = 0
			}
			reputation.Score = weightedReputation(previousScore, roundScore)
		}
		reputation.Status = reputationStatus(reputation.Score)
		reputation.LastRoundID = round.RoundID
		included := true
		if round.Sequence > warmupRounds {
			included = reputation.Status != statusBlocked &&
				!(anomalous && reputation.ConsecutiveAnomalies >= consecutiveAnomalyLimit)
		}
		assessment := ClientAssessment{
			DocType: assessmentObjectType, RoundID: round.RoundID, ExperimentID: round.ExperimentID,
			ClientID: record.ClientID, Distance: distances[index], Threshold: threshold,
			ComparableValues: comparable[index], Assessed: assessed, Anomalous: anomalous,
			Included: included, PreviousScore: previousScore, RoundScore: roundScore,
			NewScore: reputation.Score, Status: reputation.Status,
			ConsecutiveAnomalies: reputation.ConsecutiveAnomalies,
		}
		assessments[index] = assessment
		reputations[index] = reputation
	}

	report := &ReputationReport{
		DocType: reputationReportObjectType, RoundID: round.RoundID,
		ExperimentID: round.ExperimentID, Sequence: round.Sequence,
		Warmup: round.Sequence <= warmupRounds, DetectionUsed: detectionUsed,
		MedianDistance: medianDistance, MAD: mad, Threshold: threshold,
		Assessments: assessments,
	}
	return assessments, reputations, report, nil
}

func robustReferences(round *Round, records []PrototypeRecord) ([]int64, []bool) {
	references := make([]int64, round.NumClasses*round.Dimension)
	eligible := make([]bool, round.NumClasses)
	for classID := 0; classID < round.NumClasses; classID++ {
		contributors := 0
		for _, record := range records {
			if record.Counts[classID] > 0 {
				contributors++
			}
		}
		eligible[classID] = contributors >= minimumDetectorClients
		if contributors == 0 {
			continue
		}
		offset := classID * round.Dimension
		for dimensionID := 0; dimensionID < round.Dimension; dimensionID++ {
			values := make([]int64, 0, contributors)
			for _, record := range records {
				if record.Counts[classID] > 0 {
					values = append(values, record.Values[offset+dimensionID])
				}
			}
			references[offset+dimensionID] = medianInt64(values)
		}
	}
	return references, eligible
}

func loadClientReputation(ctx contractapi.TransactionContextInterface, experimentID int, clientID int) (ClientReputation, error) {
	key, err := clientReputationKey(ctx, experimentID, clientID)
	if err != nil {
		return ClientReputation{}, err
	}
	value, err := ctx.GetStub().GetState(key)
	if err != nil {
		return ClientReputation{}, fmt.Errorf("read reputation for experiment %d client %d: %w", experimentID, clientID, err)
	}
	if value == nil {
		return ClientReputation{
			DocType: reputationObjectType, ExperimentID: experimentID, ClientID: clientID,
			Score: initialReputation, Status: statusTrusted,
		}, nil
	}
	var reputation ClientReputation
	if err := json.Unmarshal(value, &reputation); err != nil {
		return ClientReputation{}, fmt.Errorf("decode reputation for experiment %d client %d: %w", experimentID, clientID, err)
	}
	return reputation, nil
}

func medianInt64(values []int64) int64 {
	if len(values) == 0 {
		return 0
	}
	sorted := append([]int64(nil), values...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
	middle := len(sorted) / 2
	if len(sorted)%2 == 1 {
		return sorted[middle]
	}
	left, right := sorted[middle-1], sorted[middle]
	return left/2 + right/2 + (left%2+right%2)/2
}

func absoluteDifference(left int64, right int64) int64 {
	if left < right {
		left, right = right, left
	}
	if right < 0 && left > math.MaxInt64+right {
		return math.MaxInt64
	}
	return left - right
}

func saturatingMultiply(value int64, multiplier int64) int64 {
	if value > 0 && multiplier > math.MaxInt64/value {
		return math.MaxInt64
	}
	return value * multiplier
}

func saturatingAdd(left int64, right int64) int64 {
	if right > math.MaxInt64-left {
		return math.MaxInt64
	}
	return left + right
}

func weightedReputation(previous int64, current int64) int64 {
	return (previous*previousScoreWeight + current*(100-previousScoreWeight) + 50) / 100
}

func reputationStatus(score int64) string {
	if score >= trustedReputation {
		return statusTrusted
	}
	if score >= blockedReputation {
		return statusWatch
	}
	return statusBlocked
}
