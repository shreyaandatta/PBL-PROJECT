// ==========================
// DATASET (REALISTIC)
// ==========================
const realBase = [
"iran israel ceasefire talks ongoing",
"us government releases economic report",
"united nations reports conflict escalation",
"scientists publish new research findings",
"official statement released by ministry",
"global leaders meet for climate summit",
"military operations reported in region",
"foreign ministers discuss peace negotiations",
"economic growth data released by government",
"health officials report new study results"
];

const fakeBase = [
"aliens secretly control world leaders",
"miracle cure doctors hate this trick",
"5g towers spreading virus conspiracy",
"secret organization running the world",
"shocking truth media hiding from you",
"cure cancer instantly with one trick",
"government hiding alien technology",
"mind control signals through phones",
"ancient aliens built modern cities",
"doctors hate this miracle discovery"
];

// expand dataset
const dataset = [];
for(let i=0;i<15;i++){
    realBase.forEach(t=>dataset.push({text:t+" "+i,label:0}));
    fakeBase.forEach(t=>dataset.push({text:t+" "+i,label:1}));
}

// ==========================
// TOKENIZATION + BIGRAM
// ==========================
function tokenize(text){
    return text.toLowerCase().split(/\W+/).filter(w=>w.length>2);
}

function generateBigrams(tokens){
    let bigrams=[];
    for(let i=0;i<tokens.length-1;i++){
        bigrams.push(tokens[i]+"_"+tokens[i+1]);
    }
    return bigrams;
}

// ==========================
// VOCAB + TFIDF
// ==========================
let vocab=[], idf=[];

function buildVocab(){
    let set=new Set();

    dataset.forEach(d=>{
        let tokens=tokenize(d.text);
        let bigrams=generateBigrams(tokens);
        [...tokens,...bigrams].forEach(w=>set.add(w));
    });

    vocab=[...set];
}

function computeIDF(){
    idf=vocab.map(w=>{
        let count=dataset.filter(d=>{
            let t=tokenize(d.text);
            let b=generateBigrams(t);
            return [...t,...b].includes(w);
        }).length;

        return Math.log((dataset.length+1)/(count+1))+1;
    });
}

function vectorize(text){
    let tokens=tokenize(text);
    let bigrams=generateBigrams(tokens);
    let features=[...tokens,...bigrams];

    let vec=new Array(vocab.length).fill(0);

    features.forEach(w=>{
        let i=vocab.indexOf(w);
        if(i!==-1) vec[i]++;
    });

    let max=Math.max(...vec,1);
    return vec.map((v,i)=>v/max*idf[i]);
}

// ==========================
// MODEL (LOGISTIC NN)
// ==========================
let W=[], bias=0;

function initModel(){
    W=new Array(vocab.length).fill(0).map(()=>Math.random()*0.1);
    bias=Math.random()*0.1;
}

function sigmoid(x){
    return 1/(1+Math.exp(-x));
}

function predict(vec){
    let sum=bias;
    for(let i=0;i<vec.length;i++){
        sum+=vec[i]*W[i];
    }
    return sigmoid(sum);
}

// ==========================
// TRAINING
// ==========================
let accuracy=0, loss=0, history=[];
let TP=0,FP=0,TN=0,FN=0;

function train(epochs=40){
    for(let e=0;e<epochs;e++){
        let correct=0,totalLoss=0;

        dataset.forEach(d=>{
            let x=vectorize(d.text);
            let y=d.label;

            let yhat=predict(x);
            let err=yhat-y;

            // gradient descent
            for(let i=0;i<W.length;i++){
                W[i]-=0.1*err*x[i];
            }
            bias-=0.1*err;

            totalLoss+=err*err;

            if((yhat>0.5?1:0)===y) correct++;
        });

        accuracy=correct/dataset.length;
        loss=totalLoss/dataset.length;
        history.push(loss);
    }

    computeConfusion();
}

// ==========================
// CONFUSION MATRIX
// ==========================
function computeConfusion(){
    TP=FP=TN=FN=0;

    dataset.forEach(d=>{
        let pred=predict(vectorize(d.text))>0.5?1:0;

        if(pred===1 && d.label===1) TP++;
        else if(pred===1 && d.label===0) FP++;
        else if(pred===0 && d.label===0) TN++;
        else FN++;
    });
}

// ==========================
// ANALYZE (FINAL LOGIC)
// ==========================
function analyzeText(text){

    let tokens=tokenize(text);

    if(tokens.length<3){
        return {verdict:"FAKE",confidence:95,score:1};
    }

    let known=tokens.filter(w=>vocab.includes(w));
    let unknownRatio=1-(known.length/tokens.length);

    if(unknownRatio>0.8){
        return {verdict:"FAKE",confidence:90,score:1};
    }

    let x=vectorize(text);
    let s=predict(x);

    // context boost (real news)
    const newsWords=["war","government","official","report","talks","minister","conflict","ceasefire","economy"];
    let newsScore=tokens.filter(w=>newsWords.includes(w)).length;

    if(newsScore>=2) s-=0.15;

    let verdict;
    if(s>0.6) verdict="FAKE";
    else if(s<0.4) verdict="REAL";
    else verdict="UNCERTAIN";

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
