package traffic

import "sync/atomic"

type Snapshot struct {
	RXBytes uint64 `json:"rx_bytes"`
	TXBytes uint64 `json:"tx_bytes"`
}

func (s Snapshot) TotalBytes() uint64 {
	return s.RXBytes + s.TXBytes
}

type Counters struct {
	rxBytes atomic.Uint64
	txBytes atomic.Uint64
}

func (c *Counters) AddRX(bytes uint64) {
	c.rxBytes.Add(bytes)
}

func (c *Counters) AddTX(bytes uint64) {
	c.txBytes.Add(bytes)
}

func (c *Counters) Snapshot() Snapshot {
	return Snapshot{
		RXBytes: c.rxBytes.Load(),
		TXBytes: c.txBytes.Load(),
	}
}
