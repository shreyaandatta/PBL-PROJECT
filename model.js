// ==========================
// LARGE DATASET (120+)
// ==========================
const dataset = [];

const realSamples = [
"NASA confirms water ice on moon",
"Scientists publish climate change report",
"Government releases economic data",
"University study shows new findings",
"Researchers discover new protein structure"
];

const fakeSamples = [
"Aliens secretly contacted leaders",
"Miracle cure doctors hate",
"5G spreads virus",
"Secret government mind control",
"Shocking conspiracy revealed"
];

// expand dataset
for(let i=0;i<25;i++){
    realSamples.forEach(t=>dataset.push({text:t+" "+i,label:0}));
    fakeSamples.forEach(t=>dataset.push({text:t+" "+i,label:1}));
}

// ==========================
// TOKENIZATION
// ==========================
function tokenize(t){
    return t.toLowerCase().split(/\W+/).filter(w=>w.length>2);
}

// ==========================
// VOCAB + TFIDF
// ==========================
let vocab=[], idf=[];

function buildVocab(){
    let set=new Set();
    dataset.forEach(d=>tokenize(d.text).forEach(w=>set.add(w)));
    vocab=[...set];
}

function computeIDF(){
    idf=vocab.map(w=>{
        let count=dataset.filter(d=>tokenize(d.text).includes(w)).length;
        return Math.log((dataset.length+1)/(count+1))+1;
    });
}

function vectorize(text){
    let vec=new Array(vocab.length).fill(0);
    let tokens=tokenize(text);

    tokens.forEach(w=>{
        let i=vocab.indexOf(w);
        if(i!==-1) vec[i]++;
    });

    let max=Math.max(...vec,1);
    return vec.map((v,i)=>v/max*idf[i]);
}

// ==========================
// 2-LAYER NEURAL NETWORK
// ==========================
let W1=[], W2=[], b1=[], b2;

function initModel(){
    let input=vocab.length, hidden=16;

    W1=Array.from({length:input},()=>Array.from({length:hidden},()=>Math.random()-0.5));
    W2=Array.from({length:hidden},()=>Math.random()-0.5);

    b1=new Array(hidden).fill(0);
    b2=0;
}

function relu(x){return Math.max(0,x);}
function sigmoid(x){return 1/(1+Math.exp(-x));}

// forward
function forward(x){
    let h=[];

    for(let j=0;j<W2.length;j++){
        let sum=b1[j];
        for(let i=0;i<x.length;i++){
            sum+=x[i]*W1[i][j];
        }
        h.push(relu(sum));
    }

    let out=b2;
    for(let j=0;j<h.length;j++){
        out+=h[j]*W2[j];
    }

    return sigmoid(out);
}

// ==========================
// TRAINING + METRICS
// ==========================
let accuracy=0, loss=0;
let history=[];

let TP=0,FP=0,TN=0,FN=0;

function train(epochs=30){

    for(let e=0;e<epochs;e++){
        let correct=0,totalLoss=0;

        dataset.forEach(d=>{
            let x=vectorize(d.text);
            let y=d.label;

            let yhat=forward(x);
            let err=yhat-y;

            totalLoss+=err*err;

            // update W2
            for(let j=0;j<W2.length;j++){
                W2[j]-=0.05*err;
            }

            b2-=0.05*err;

            if((yhat>0.5?1:0)===y) correct++;
        });

        accuracy=correct/dataset.length;
        loss=totalLoss/dataset.length;
        history.push(loss);
    }

    computeConfusion();
}

function computeConfusion(){
    TP=FP=TN=FN=0;

    dataset.forEach(d=>{
        let pred=forward(vectorize(d.text))>0.5?1:0;

        if(pred===1 && d.label===1) TP++;
        else if(pred===1 && d.label===0) FP++;
        else if(pred===0 && d.label===0) TN++;
        else FN++;
    });
}

// ==========================
// FUZZY LOGIC
// ==========================
function fuzzy(text,score){
    if(text.length<20) score+=0.2;
    if(/[!]{2,}/.test(text)) score+=0.2;
    return score;
}

// ==========================
// ANALYZE
// ==========================
function analyzeText(text){

    const tokens = tokenize(text);

    // 🚨 NEW: UNKNOWN WORD DETECTION
    let known = tokens.filter(w => vocab.includes(w));

    if (tokens.length < 3 || known.length / tokens.length < 0.3) {
        return {
            verdict: "FAKE",
            confidence: 95,
            score: 1,
            reason: "Text does not match learned linguistic patterns (out-of-vocabulary)"
        };
    }
    if(text.split(" ").length<3){
        return {verdict:"FAKE",confidence:95,score:1};
    }

    let x=vectorize(text);
    let s=forward(x);
    s=fuzzy(text,s);

    let verdict=s>0.5?"FAKE":"REAL";

    return {
        verdict,
        confidence:Math.round(60+Math.abs(s-0.5)*80),
        score:s
    };
}

// INIT
buildVocab();
computeIDF();
initModel();
train();
