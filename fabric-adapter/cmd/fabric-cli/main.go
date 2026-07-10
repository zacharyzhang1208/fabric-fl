package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"fabric-fl/fabric-adapter/internal/fabric"
	gatewayclient "github.com/hyperledger/fabric-gateway/pkg/client"
	"google.golang.org/grpc/status"
)

const usageText = `Usage:
  fabric-cli get [flags] KEY
  fabric-cli set [flags] KEY VALUE
  fabric-cli evaluate [flags] TXN [ARG...]
  fabric-cli submit [flags] TXN [ARG...]

Examples:
  fabric-cli set round:1 '{"accuracy":0.91}'
  fabric-cli get round:1
  fabric-cli submit Set hello world
  fabric-cli evaluate Get hello
`

func main() {
	log.SetFlags(0)

	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}

	command := os.Args[1]
	if command == "help" || command == "-h" || command == "--help" {
		usage()
		return
	}

	config, args := parseFlags(command, os.Args[2:])

	client, err := fabric.Connect(config)
	if err != nil {
		fatal("connect failed", err)
	}
	defer client.Close()

	result, err := run(command, client, args)
	if err != nil {
		fatal(fmt.Sprintf("%s failed", command), err)
	}

	if len(result) > 0 {
		fmt.Println(string(result))
		return
	}

	if command == "set" || command == "submit" {
		fmt.Println("OK")
	}
}

func fatal(prefix string, err error) {
	fmt.Fprintf(os.Stderr, "%s: %v\n", prefix, err)

	switch err := err.(type) {
	case *gatewayclient.EndorseError:
		printTransactionError("endorse", err.TransactionError)
	case *gatewayclient.SubmitError:
		printTransactionError("submit", err.TransactionError)
	case *gatewayclient.CommitStatusError:
		printTransactionError("commit status", err.TransactionError)
	case *gatewayclient.CommitError:
		fmt.Fprintf(os.Stderr, "commit transaction ID: %s\n", err.TransactionID)
		fmt.Fprintf(os.Stderr, "commit status: %s\n", err.Code.String())
	}

	os.Exit(1)
}

func printTransactionError(stage string, err *gatewayclient.TransactionError) {
	fmt.Fprintf(os.Stderr, "%s transaction ID: %s\n", stage, err.TransactionID)

	status := status.Convert(err)
	fmt.Fprintf(os.Stderr, "%s gRPC code: %s\n", stage, status.Code())

	for i, detail := range status.Details() {
		fmt.Fprintf(os.Stderr, "%s detail %d: %v\n", stage, i+1, detail)
	}
}

func parseFlags(command string, args []string) (fabric.Config, []string) {
	flags := flag.NewFlagSet(command, flag.ExitOnError)

	config := fabric.Config{}
	flags.StringVar(&config.MSPID, "mspid", getenv("FABRIC_MSP_ID", "Org1MSP"), "client MSP ID")
	flags.StringVar(&config.CertPath, "cert", getenv("FABRIC_CERT_PATH", ""), "client signing certificate path")
	flags.StringVar(&config.KeyPath, "key", getenv("FABRIC_KEY_PATH", ""), "client private key path")
	flags.StringVar(&config.TLSCertPath, "tls-cert", getenv("FABRIC_TLS_CERT_PATH", ""), "peer TLS CA certificate path")
	flags.StringVar(&config.Peer, "peer", getenv("FABRIC_PEER_ENDPOINT", "localhost:7051"), "peer gateway endpoint")
	flags.StringVar(&config.PeerHost, "peer-host", getenv("FABRIC_PEER_HOST", "peer0.org1.example.com"), "peer TLS host name")
	flags.StringVar(&config.Channel, "channel", getenv("FABRIC_CHANNEL", "trainingchannel"), "channel name")
	flags.StringVar(&config.Chaincode, "chaincode", getenv("FABRIC_CHAINCODE", "contracts"), "chaincode name")
	flags.DurationVar(&config.Timeout, "timeout", getenvDuration("FABRIC_TIMEOUT", 10*time.Second), "gateway operation timeout")

	flags.Usage = usage
	if err := flags.Parse(args); err != nil {
		os.Exit(2)
	}

	return config, flags.Args()
}

func run(command string, client *fabric.Client, args []string) ([]byte, error) {
	switch command {
	case "get":
		if len(args) != 1 {
			return nil, fmt.Errorf("get expects KEY")
		}
		return client.Evaluate("Get", args[0])
	case "set":
		if len(args) != 2 {
			return nil, fmt.Errorf("set expects KEY VALUE")
		}
		return client.Submit("Set", args[0], args[1])
	case "evaluate":
		if len(args) < 1 {
			return nil, fmt.Errorf("evaluate expects TXN [ARG...]")
		}
		return client.Evaluate(args[0], args[1:]...)
	case "submit":
		if len(args) < 1 {
			return nil, fmt.Errorf("submit expects TXN [ARG...]")
		}
		return client.Submit(args[0], args[1:]...)
	default:
		return nil, fmt.Errorf("unknown command %q", command)
	}
}

func getenv(name string, fallback string) string {
	value := strings.TrimSpace(os.Getenv(name))
	if value == "" {
		return fallback
	}
	return value
}

func getenvDuration(name string, fallback time.Duration) time.Duration {
	value := strings.TrimSpace(os.Getenv(name))
	if value == "" {
		return fallback
	}

	duration, err := time.ParseDuration(value)
	if err != nil {
		log.Fatalf("invalid %s duration %q: %v", name, value, err)
	}

	return duration
}

func usage() {
	fmt.Fprint(os.Stderr, usageText)
	fmt.Fprintln(os.Stderr)
	fmt.Fprintln(os.Stderr, "Flags can be provided directly or by environment variables:")
	fmt.Fprintln(os.Stderr, "  FABRIC_MSP_ID, FABRIC_CERT_PATH, FABRIC_KEY_PATH, FABRIC_TLS_CERT_PATH")
	fmt.Fprintln(os.Stderr, "  FABRIC_PEER_ENDPOINT, FABRIC_PEER_HOST, FABRIC_CHANNEL, FABRIC_CHAINCODE, FABRIC_TIMEOUT")
}
