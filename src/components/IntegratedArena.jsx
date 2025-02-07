import React, { useState, useEffect, useCallback } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { Bluetooth, Radio, Wifi } from 'lucide-react';

// Status Indicator Component
const StatusIndicator = ({ icon, label, status }) => (
  <div className="status-indicator">
    <div className={`status-icon ${status ? 'status-icon-active' : 'status-icon-inactive'}`}>
      {icon}
    </div>
    <span>{label}</span>
  </div>
);

// Device List Component
const DeviceList = ({ devices }) => (
  <div className="card">
    <h2 className="card-title">Device Locations</h2>
    <div className="device-list">
      {devices.map(device => (
        <div key={device.id} className="device-item">
          <div className="device-info">
            <div className="device-name">{device.name}</div>
            <div className="device-seat">Seat: {device.seat_id}</div>
          </div>
          <div>
            {Object.entries(device.rssi_values).map(([sensor, value]) => (
              <div key={sensor} className="signal-value">
                {sensor}: <span className={value > -60 ? 'signal-good' : 'signal-moderate'}>
                  {value.toFixed(1)} dBm
                </span>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  </div>
);

// Arena Grid Component
const ArenaGrid = ({ devices, sectionColors }) => {
  const devicesBySeat = devices.reduce((acc, device) => {
    if (!acc[device.seat_id]) acc[device.seat_id] = [];
    acc[device.seat_id].push(device);
    return acc;
  }, {});

  return (
    <div className="card">
      <h2 className="card-title">Arena Visualization</h2>
      <div className="arena-grid">
        {Array.from({ length: 88 }).map((_, idx) => {
          const row = Math.floor(idx / 11);
          const col = idx % 11;
          const seatId = `${String.fromCharCode(65 + row)}${col + 1}`;
          const devicesInSeat = devicesBySeat[seatId] || [];
          
          return (
            <div
              key={seatId}
              className="grid-cell"
              style={{
                backgroundColor: sectionColors[seatId[0]] || '#1F2937',
                opacity: devicesInSeat.length > 0 ? 1 : 0.3,
              }}
            >
              <div className="seat-label">{seatId}</div>
              {devicesInSeat.length > 0 && (
                <div className="device-count">
                  {devicesInSeat.length}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};

// Signal Strength Chart Component
const SignalStrengthChart = ({ devices }) => {
  const chartData = devices.map(device => ({
    name: device.name,
    ...device.rssi_values
  }));

  return (
    <div className="card">
      <h2 className="card-title">Signal Strength Analysis</h2>
      <div className="chart-container">
        <ResponsiveContainer>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="name" stroke="#fff" />
            <YAxis stroke="#fff" domain={[-100, -20]} />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: '#1F2937',
                border: 'none',
                borderRadius: '8px',
                color: '#fff'
              }}
            />
            <Legend />
            <Line type="monotone" dataKey="NW" name="NW Sensor" stroke="#EF4444" strokeWidth={2} dot={{ fill: '#EF4444' }} />
            <Line type="monotone" dataKey="NE" name="NE Sensor" stroke="#10B981" strokeWidth={2} dot={{ fill: '#10B981' }} />
            <Line type="monotone" dataKey="SW" name="SW Sensor" stroke="#3B82F6" strokeWidth={2} dot={{ fill: '#3B82F6' }} />
            <Line type="monotone" dataKey="SE" name="SE Sensor" stroke="#F59E0B" strokeWidth={2} dot={{ fill: '#F59E0B' }} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

// Section Control Component
const SectionControl = ({ sectionColors, setSectionColors }) => {
  const colors = ['#1F2937', '#EF4444', '#10B981', '#3B82F6', '#F59E0B', '#8B5CF6', '#EC4899', '#6366F1'];

  return (
    <div className="card">
      <h2 className="card-title">Section Control</h2>
      <div className="section-control">
        {Object.keys(sectionColors).map(section => (
          <button
            key={section}
            onClick={() => {
              const currentIndex = colors.indexOf(sectionColors[section]);
              const nextColor = colors[(currentIndex + 1) % colors.length];
              setSectionColors(prev => ({ ...prev, [section]: nextColor }));
            }}
            className="section-button"
            style={{
              backgroundColor: sectionColors[section],
            }}
          >
            Section {section}
          </button>
        ))}
      </div>
    </div>
  );
};

const IntegratedArenaSystem = () => {
  const [devices, setDevices] = useState([]);
  const [wsConnected, setWsConnected] = useState(false);
  const [sectionColors, setSectionColors] = useState({
    'A': '#1F2937', 'B': '#1F2937', 'C': '#1F2937', 'D': '#1F2937',
    'E': '#1F2937', 'F': '#1F2937', 'G': '#1F2937', 'H': '#1F2937'
  });

  const connectWebSocket = useCallback(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.hostname}:8000/ws`;
    console.log('Connecting to WebSocket:', wsUrl);
    
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      console.log('WebSocket Connected');
      setWsConnected(true);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log('Received WebSocket data:', data);
        
        if (data.devices) {
          const transformedDevices = data.devices.map(device => ({
            id: device.id,
            name: device.name || 'Unknown Device',
            seat_id: device.location ? 
              `${String.fromCharCode(65 + Math.floor(device.location[0]))}${Math.floor(device.location[1]) + 1}` : 
              'Unknown',
            rssi_values: {
              NW: device.rssi_values?.NW || device.rssi_values?.scanner || -100,
              NE: device.rssi_values?.NE || -100,
              SW: device.rssi_values?.SW || -100,
              SE: device.rssi_values?.SE || -100
            }
          }));
          console.log('Transformed devices:', transformedDevices);
          setDevices(transformedDevices);
        }
      } catch (error) {
        console.error('Error parsing WebSocket message:', error, event.data);
      }
    };

    ws.onclose = (event) => {
      console.log('WebSocket Disconnected:', event.code, event.reason);
      setWsConnected(false);
      setTimeout(connectWebSocket, 2000);
    };

    ws.onerror = (error) => {
      console.error('WebSocket Error:', error);
    };

    return () => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
    };
  }, []);

  useEffect(() => {
    const cleanup = connectWebSocket();
    return () => {
      if (cleanup) cleanup();
    };
  }, [connectWebSocket]);

  return (
    <div className="container">
      <header className="header">
        <h1 className="header-title">Arena Interactive System</h1>
        <div className="status-container">
          <StatusIndicator 
            icon={<Radio size={24} />} 
            label="Pico Network" 
            status={wsConnected} 
          />
          <StatusIndicator 
            icon={<Bluetooth size={24} />} 
            label="Bluetooth Scanner" 
            status={devices.length > 0} 
          />
          <StatusIndicator 
            icon={<Wifi size={24} />} 
            label="Server Connection" 
            status={wsConnected} 
          />
        </div>
      </header>

      <div className="grid">
        <DeviceList devices={devices} />
        <ArenaGrid devices={devices} sectionColors={sectionColors} />
        <SignalStrengthChart devices={devices} />
        <SectionControl sectionColors={sectionColors} setSectionColors={setSectionColors} />
      </div>
    </div>
  );
};

export default IntegratedArenaSystem;