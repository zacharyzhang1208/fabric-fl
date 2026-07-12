package main

import (
	"math"
	"testing"

	"github.com/hyperledger/fabric-contract-api-go/v2/contractapi"
)

func TestContractMetadataBuilds(t *testing.T) {
	chaincode, err := contractapi.NewChaincode(&SmartContract{})
	if err != nil {
		t.Fatalf("NewChaincode() error = %v", err)
	}
	if chaincode == nil {
		t.Fatal("NewChaincode() returned nil")
	}
}

func TestAggregatePrototypesUsesEqualClientWeightPerClass(t *testing.T) {
	round := &Round{
		RoundID:         1,
		ExpectedClients: 2,
		NumClasses:      2,
		Dimension:       2,
		Scale:           100,
	}
	records := []PrototypeRecord{
		{PrototypePayload: PrototypePayload{
			Values: []int64{100, 200, 300, 400},
			Counts: []int64{5, 0},
		}},
		{PrototypePayload: PrototypePayload{
			Values: []int64{200, 400, 500, 700},
			Counts: []int64{2, 1},
		}},
	}

	global, err := aggregatePrototypes(round, records)
	if err != nil {
		t.Fatalf("aggregatePrototypes() error = %v", err)
	}
	wantValues := []int64{150, 300, 500, 700}
	wantCounts := []int64{2, 1}
	assertInt64Slice(t, global.Values, wantValues)
	assertInt64Slice(t, global.Counts, wantCounts)
}

func TestAggregatePrototypesRoundsHalfAwayFromZero(t *testing.T) {
	round := &Round{RoundID: 1, ExpectedClients: 2, NumClasses: 1, Dimension: 2, Scale: 1}
	records := []PrototypeRecord{
		{PrototypePayload: PrototypePayload{Values: []int64{0, 0}, Counts: []int64{1}}},
		{PrototypePayload: PrototypePayload{Values: []int64{1, -1}, Counts: []int64{1}}},
	}

	global, err := aggregatePrototypes(round, records)
	if err != nil {
		t.Fatalf("aggregatePrototypes() error = %v", err)
	}
	assertInt64Slice(t, global.Values, []int64{1, -1})
}

func TestAggregatePrototypesRejectsOverflow(t *testing.T) {
	round := &Round{RoundID: 1, ExpectedClients: 2, NumClasses: 1, Dimension: 1, Scale: 1}
	records := []PrototypeRecord{
		{PrototypePayload: PrototypePayload{Values: []int64{math.MaxInt64}, Counts: []int64{1}}},
		{PrototypePayload: PrototypePayload{Values: []int64{1}, Counts: []int64{1}}},
	}

	if _, err := aggregatePrototypes(round, records); err == nil {
		t.Fatal("aggregatePrototypes() error = nil, want overflow error")
	}
}

func TestValidatePrototypePayload(t *testing.T) {
	round := &Round{RoundID: 3, ExpectedClients: 2, NumClasses: 2, Dimension: 2, Scale: 1000}
	payload := PrototypePayload{
		Encoding: prototypeEncoding,
		RoundID:  3,
		ClientID: 1,
		Shape:    []int{2, 2},
		Scale:    1000,
		Values:   []int64{1, 2, 3, 4},
		Counts:   []int64{2, 0},
	}

	if err := validatePrototypePayload(payload, round, 1); err != nil {
		t.Fatalf("validatePrototypePayload() error = %v", err)
	}
	payload.Shape = []int{2, 3}
	if err := validatePrototypePayload(payload, round, 1); err == nil {
		t.Fatal("validatePrototypePayload() error = nil, want shape error")
	}
}

func assertInt64Slice(t *testing.T, got []int64, want []int64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("len(got) = %d, want %d", len(got), len(want))
	}
	for index := range want {
		if got[index] != want[index] {
			t.Fatalf("got[%d] = %d, want %d", index, got[index], want[index])
		}
	}
}
