package main

import (
	"context"
	"errors"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"fabric-fl/fabric-adapter/internal/fabric"
	"fabric-fl/fabric-adapter/internal/httpapi"
)

const shutdownTimeout = 10 * time.Second

func main() {
	config := fabricConfigFromEnv()
	client, err := fabric.Connect(config)
	if err != nil {
		log.Fatalf("connect to Fabric: %v", err)
	}
	defer client.Close()

	address := getenv("FABRIC_ADAPTER_ADDRESS", "127.0.0.1:18080")
	server := &http.Server{
		Addr:              address,
		Handler:           httpapi.New(client),
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       15 * time.Second,
		WriteTimeout:      30 * time.Second,
		IdleTimeout:       60 * time.Second,
	}

	errCh := make(chan error, 1)
	go func() {
		log.Printf("Fabric adapter listening on http://%s", address)
		if err := server.ListenAndServe(); !errors.Is(err, http.ErrServerClosed) {
			errCh <- err
			return
		}
		errCh <- nil
	}()

	signalCh := make(chan os.Signal, 1)
	signal.Notify(signalCh, os.Interrupt, syscall.SIGTERM)
	defer signal.Stop(signalCh)

	select {
	case err := <-errCh:
		if err != nil {
			log.Fatalf("serve HTTP: %v", err)
		}
		return
	case sig := <-signalCh:
		log.Printf("received %s, shutting down", sig)
	}

	ctx, cancel := context.WithTimeout(context.Background(), shutdownTimeout)
	defer cancel()
	if err := server.Shutdown(ctx); err != nil {
		log.Printf("HTTP shutdown: %v", err)
	}
	if err := <-errCh; err != nil {
		log.Printf("HTTP server stopped: %v", err)
	}
}

func fabricConfigFromEnv() fabric.Config {
	return fabric.Config{
		MSPID:       getenv("FABRIC_MSP_ID", "Org1MSP"),
		CertPath:    getenv("FABRIC_CERT_PATH", ""),
		KeyPath:     getenv("FABRIC_KEY_PATH", ""),
		TLSCertPath: getenv("FABRIC_TLS_CERT_PATH", ""),
		Peer:        getenv("FABRIC_PEER_ENDPOINT", "localhost:7051"),
		PeerHost:    getenv("FABRIC_PEER_HOST", "peer0.org1.example.com"),
		Channel:     getenv("FABRIC_CHANNEL", "trainingchannel"),
		Chaincode:   getenv("FABRIC_CHAINCODE", "contracts"),
		Timeout:     getenvDuration("FABRIC_TIMEOUT", 10*time.Second),
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
