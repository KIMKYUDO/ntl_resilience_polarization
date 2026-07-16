document.addEventListener("DOMContentLoaded", async function () {

    // =====================================================
    // 1. LOAD PROJECT OVERVIEW FROM FLASK BACKEND
    // =====================================================

    try {
        const response = await fetch("/api/overview");
        const data = await response.json();

        document.getElementById("eventCount").textContent = data.events;
        document.getElementById("primaryModel").textContent = data.primary_model;
        document.getElementById("auroc").textContent =
            (data.auroc * 100).toFixed(2) + "%";
        document.getElementById("recall").textContent =
            (data.top30_recall * 100).toFixed(2) + "%";

    } catch (error) {
        console.error("Could not connect to backend:", error);
    }


    // =====================================================
    // 2. INITIALIZE INDIA MAP
    // =====================================================

    const map = L.map("map").setView([22.5, 79.0], 5);

    L.tileLayer(
        "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        {
            attribution: "&copy; OpenStreetMap contributors"
        }
    ).addTo(map);


    // =====================================================
    // 3. DEMO EVENT DATA
    // Replace with real model outputs later.
    // =====================================================

    const eventData = {

        "Cyclone Amphan (2020)": {
            name: "Cyclone Amphan",
            location: "West Bengal",
            lat: 22.57,
            lon: 88.36,
            risk: 87,
            trend: "Slow Recovery",
            risk12: 18,
            risk24: 8
        },

        "Cyclone Nivar (2020)": {
            name: "Cyclone Nivar",
            location: "Tamil Nadu",
            lat: 12.91,
            lon: 79.13,
            risk: 81,
            trend: "Slow Recovery",
            risk12: 14,
            risk24: 6
        },

        "Cyclone Tauktae (2021)": {
            name: "Cyclone Tauktae",
            location: "Western India",
            lat: 20.59,
            lon: 72.93,
            risk: 74,
            trend: "Moderate Recovery",
            risk12: 12,
            risk24: 5
        },

        "Cyclone Yaas (2021)": {
            name: "Cyclone Yaas",
            location: "Odisha / West Bengal",
            lat: 21.64,
            lon: 87.48,
            risk: 79,
            trend: "Slow Recovery",
            risk12: 15,
            risk24: 7
        },

        "Cyclone Fani (2019)": {
            name: "Cyclone Fani",
            location: "Odisha",
            lat: 20.29,
            lon: 85.82,
            risk: 72,
            trend: "Moderate Recovery",
            risk12: 11,
            risk24: 4
        },

        "Hyderabad Flood (2020)": {
            name: "Hyderabad Urban Flood",
            location: "Hyderabad",
            lat: 17.38,
            lon: 78.48,
            risk: 69,
            trend: "Moderate Recovery",
            risk12: 10,
            risk24: 4
        },

        "Chennai Flood (2021)": {
            name: "Chennai Urban Flood",
            location: "Chennai",
            lat: 13.08,
            lon: 80.27,
            risk: 76,
            trend: "Slow Recovery",
            risk12: 13,
            risk24: 5
        }

    };


    // =====================================================
    // 4. RISK COLOR
    // =====================================================

    function getRiskColor(risk) {

        if (risk >= 80) {
            return "#e76f51";
        }

        if (risk >= 60) {
            return "#f4a261";
        }

        return "#4fd1c5";
    }


    function getRiskCategory(risk) {

        if (risk >= 80) {
            return "HIGH RISK";
        }

        if (risk >= 60) {
            return "MODERATE RISK";
        }

        return "LOW RISK";
    }


    // =====================================================
    // 5. ADD EVENT MARKERS
    // =====================================================

    const markers = {};

    Object.entries(eventData).forEach(([eventName, event]) => {

        const marker = L.circleMarker(
            [event.lat, event.lon],
            {
                radius: 10,
                fillColor: getRiskColor(event.risk),
                color: "#ffffff",
                weight: 1,
                fillOpacity: 0.85
            }
        ).addTo(map);


        marker.bindPopup(`
            <strong>${event.name}</strong>
            <br>
            ${event.location}
            <br><br>
            Demo Recovery Risk:
            <strong>${event.risk}%</strong>
        `);


        markers[eventName] = marker;

    });


    // =====================================================
    // 6. ANALYZE RECOVERY
    // =====================================================

    const analyzeButton =
        document.getElementById("analyzeButton");


    analyzeButton.addEventListener("click", function () {

        const selectedEvent =
            document.getElementById("analysisEvent").value;

        const result =
            document.getElementById("analysisResult");


        if (selectedEvent === "Select Disaster Event") {

            result.innerHTML = `
                <p class="analysis-warning">
                    Please select a disaster event.
                </p>
            `;

            return;
        }


        const event = eventData[selectedEvent];


        if (!event) {

            result.innerHTML = `
                <p>
                    Event data is currently unavailable.
                </p>
            `;

            return;
        }


        // Move map to selected event

        map.flyTo(
            [event.lat, event.lon],
            7,
            {
                duration: 1.5
            }
        );


        // Open corresponding marker

        if (markers[selectedEvent]) {
            markers[selectedEvent].openPopup();
        }


        // Display dynamic analysis

        result.innerHTML = `

            <div class="recovery-result">

                <div class="result-header">

                    <div>
                        <span class="section-label">
                            DEMO RECOVERY ANALYSIS
                        </span>

                        <h2>
                            ${event.name}
                        </h2>

                        <p>
                            ${event.location}
                        </p>
                    </div>

                    <div
                        class="result-risk"
                        style="
                            border-color:
                            ${getRiskColor(event.risk)}
                        "
                    >

                        <span>
                            ${getRiskCategory(event.risk)}
                        </span>

                        <strong>
                            ${event.risk}%
                        </strong>

                    </div>

                </div>


                <div class="result-metrics">

                    <div class="result-metric">

                        <span>
                            Recovery Trend
                        </span>

                        <strong>
                            ${event.trend}
                        </strong>

                    </div>


                    <div class="result-metric">

                        <span>
                            12-Month No-Recovery Risk
                        </span>

                        <strong>
                            ${event.risk12}%
                        </strong>

                    </div>


                    <div class="result-metric">

                        <span>
                            24-Month No-Recovery Risk
                        </span>

                        <strong>
                            ${event.risk24}%
                        </strong>

                    </div>


                    <div class="result-metric">

                        <span>
                            Analysis Window
                        </span>

                        <strong>
                            3 Months
                        </strong>

                    </div>

                </div>


                <div class="demo-notice">

                    <i class="fa-solid fa-circle-info"></i>

                    Demo values are currently used for
                    frontend development. These will be
                    replaced by actual model predictions.

                </div>

            </div>

        `;


        // Scroll map into view after analysis

        document
            .getElementById("risk-map")
            .scrollIntoView({
                behavior: "smooth"
            });

    });

});