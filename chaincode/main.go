package main

import (
	"fmt"

	"github.com/hyperledger/fabric-contract-api-go/v2/contractapi"
)

type SmartContract struct {
	contractapi.Contract
}

func (s *SmartContract) Set(ctx contractapi.TransactionContextInterface, key string, value string) error {
	return ctx.GetStub().PutState(key, []byte(value))
}

func (s *SmartContract) Get(ctx contractapi.TransactionContextInterface, key string) (string, error) {
	value, err := ctx.GetStub().GetState(key)
	if err != nil {
		return "", fmt.Errorf("failed to read key %q: %w", key, err)
	}
	if value == nil {
		return "", fmt.Errorf("key %q does not exist", key)
	}

	return string(value), nil
}

func main() {
	chaincode, err := contractapi.NewChaincode(&SmartContract{})
	if err != nil {
		panic(fmt.Errorf("failed to create chaincode: %w", err))
	}

	if err := chaincode.Start(); err != nil {
		panic(fmt.Errorf("failed to start chaincode: %w", err))
	}
}
