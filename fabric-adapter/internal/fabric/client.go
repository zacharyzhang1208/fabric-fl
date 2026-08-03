package fabric

import (
	"context"
	"crypto/x509"
	"fmt"
	"os"
	"time"

	"fabric-fl/fabric-adapter/internal/traffic"
	"github.com/hyperledger/fabric-gateway/pkg/client"
	"github.com/hyperledger/fabric-gateway/pkg/identity"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/stats"
)

type Config struct {
	MSPID       string
	CertPath    string
	KeyPath     string
	TLSCertPath string
	Peer        string
	PeerHost    string
	Channel     string
	Chaincode   string
	Timeout     time.Duration
}

type Client struct {
	gateway *client.Gateway
	conn    *grpc.ClientConn
	config  Config
	traffic *traffic.Counters
}

func Connect(config Config) (*Client, error) {
	if err := validateConfig(config); err != nil {
		return nil, err
	}

	id, err := newIdentity(config.MSPID, config.CertPath)
	if err != nil {
		return nil, fmt.Errorf("load identity: %w", err)
	}

	sign, err := newSign(config.KeyPath)
	if err != nil {
		return nil, fmt.Errorf("load signer: %w", err)
	}

	grpcTraffic := &traffic.Counters{}
	conn, err := newGrpcConnection(
		config.Peer,
		config.PeerHost,
		config.TLSCertPath,
		config.Timeout,
		grpcTraffic,
	)
	if err != nil {
		return nil, fmt.Errorf("connect peer: %w", err)
	}

	gateway, err := client.Connect(
		id,
		client.WithSign(sign),
		client.WithClientConnection(conn),
		client.WithEvaluateTimeout(config.Timeout),
		client.WithEndorseTimeout(config.Timeout),
		client.WithSubmitTimeout(config.Timeout),
		client.WithCommitStatusTimeout(config.Timeout),
	)
	if err != nil {
		_ = conn.Close()
		return nil, fmt.Errorf("connect gateway: %w", err)
	}

	return &Client{
		gateway: gateway,
		conn:    conn,
		config:  config,
		traffic: grpcTraffic,
	}, nil
}

func (c *Client) TrafficSnapshot() traffic.Snapshot {
	return c.traffic.Snapshot()
}

func (c *Client) Close() {
	if c.gateway != nil {
		c.gateway.Close()
	}
	if c.conn != nil {
		_ = c.conn.Close()
	}
}

func (c *Client) Evaluate(transaction string, args ...string) ([]byte, error) {
	contract := c.gateway.GetNetwork(c.config.Channel).GetContract(c.config.Chaincode)
	return contract.EvaluateTransaction(transaction, args...)
}

func (c *Client) Submit(transaction string, args ...string) ([]byte, error) {
	contract := c.gateway.GetNetwork(c.config.Channel).GetContract(c.config.Chaincode)
	return contract.SubmitTransaction(transaction, args...)
}

func validateConfig(config Config) error {
	required := map[string]string{
		"mspid":     config.MSPID,
		"cert":      config.CertPath,
		"key":       config.KeyPath,
		"tls-cert":  config.TLSCertPath,
		"peer":      config.Peer,
		"peer-host": config.PeerHost,
		"channel":   config.Channel,
		"chaincode": config.Chaincode,
	}

	for name, value := range required {
		if value == "" {
			return fmt.Errorf("%s is required", name)
		}
	}

	if config.Timeout <= 0 {
		return fmt.Errorf("timeout must be positive")
	}

	return nil
}

func newIdentity(mspID string, certPath string) (*identity.X509Identity, error) {
	certPEM, err := os.ReadFile(certPath)
	if err != nil {
		return nil, err
	}

	cert, err := identity.CertificateFromPEM(certPEM)
	if err != nil {
		return nil, err
	}

	return identity.NewX509Identity(mspID, cert)
}

func newSign(keyPath string) (identity.Sign, error) {
	keyPEM, err := os.ReadFile(keyPath)
	if err != nil {
		return nil, err
	}

	privateKey, err := identity.PrivateKeyFromPEM(keyPEM)
	if err != nil {
		return nil, err
	}

	return identity.NewPrivateKeySign(privateKey)
}

func newGrpcConnection(
	peer string,
	peerHost string,
	tlsCertPath string,
	timeout time.Duration,
	counters *traffic.Counters,
) (*grpc.ClientConn, error) {
	tlsPEM, err := os.ReadFile(tlsCertPath)
	if err != nil {
		return nil, err
	}

	certPool := x509.NewCertPool()
	if !certPool.AppendCertsFromPEM(tlsPEM) {
		return nil, fmt.Errorf("failed to add TLS certificate to pool")
	}

	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	return grpc.DialContext(
		ctx,
		peer,
		grpc.WithTransportCredentials(credentials.NewClientTLSFromCert(certPool, peerHost)),
		grpc.WithStatsHandler(&grpcTrafficHandler{counters: counters}),
		grpc.WithBlock(),
	)
}

type grpcTrafficHandler struct {
	counters *traffic.Counters
}

func (h *grpcTrafficHandler) TagRPC(ctx context.Context, _ *stats.RPCTagInfo) context.Context {
	return ctx
}

func (h *grpcTrafficHandler) HandleRPC(_ context.Context, rpcStats stats.RPCStats) {
	switch value := rpcStats.(type) {
	case *stats.InPayload:
		h.addRX(value.WireLength)
	case *stats.InHeader:
		h.addRX(value.WireLength)
	case *stats.InTrailer:
		h.addRX(value.WireLength)
	case *stats.OutPayload:
		h.addTX(value.WireLength)
	}
}

func (h *grpcTrafficHandler) TagConn(ctx context.Context, _ *stats.ConnTagInfo) context.Context {
	return ctx
}

func (h *grpcTrafficHandler) HandleConn(context.Context, stats.ConnStats) {}

func (h *grpcTrafficHandler) addRX(bytes int) {
	if bytes > 0 {
		h.counters.AddRX(uint64(bytes))
	}
}

func (h *grpcTrafficHandler) addTX(bytes int) {
	if bytes > 0 {
		h.counters.AddTX(uint64(bytes))
	}
}
